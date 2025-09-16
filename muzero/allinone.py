import jax
import jax.numpy as jnp
import chex
from typing import Callable, Tuple, Optional, Any, ClassVar


# ------------------------------
# 1. 核心依赖与类型定义
# ------------------------------

# 模型参数类型（任意JAX数组结构）
Params = chex.ArrayTree

# 动作类型（整数数组）
Action = chex.Array

# 递归网络输出（状态转移后的预测结果）
@chex.dataclass(frozen=True)
class RecurrentFnOutput:
    reward: chex.Array  # [B] 动作奖励
    discount: chex.Array  # [B] 折扣因子
    prior_logits: chex.Array  # [B, num_actions] 动作先验logits
    value: chex.Array  # [B] 下一状态价值

# 表示网络输出（根节点初始预测）
@chex.dataclass(frozen=True)
class RootFnOutput:
    prior_logits: chex.Array  # [B, num_actions] 根节点动作先验
    value: chex.Array  # [B] 根节点价值
    embedding: Any  # [B, ...] 根节点状态嵌入

# 策略输出（搜索后的决策结果）
@chex.dataclass(frozen=True)
class PolicyOutput:
    action: chex.Array  # [B] 选中的动作
    action_weights: chex.Array  # [B, num_actions] 动作权重（用于训练）
    search_tree: Any  # 搜索树（用于调试/分析）

# 搜索摘要（提取关键搜索结果）
@chex.dataclass(frozen=True)
class SearchSummary:
    visit_counts: chex.Array  # [B, num_actions] 动作访问次数
    visit_probs: chex.Array  # [B, num_actions] 动作访问概率
    value: chex.Array  # [B] 根节点价值
    qvalues: chex.Array  # [B, num_actions] 动作Q值

# 递归网络函数类型：(参数, 随机键, 动作, 嵌入) → (输出, 新嵌入)
RecurrentFn = Callable[[Params, chex.PRNGKey, Action, Any], Tuple[RecurrentFnOutput, Any]]
# 表示网络函数类型：(参数, 随机键, 状态) → 根节点输出
RootFn = Callable[[Params, chex.PRNGKey, Any], RootFnOutput]
# Q值转换函数类型：(树, 节点索引) → 转换后的Q值
QTransform = Callable[[Any, chex.Numeric], chex.Array]


# ------------------------------
# 2. 工具函数
# ------------------------------

def masked_argmax(to_argmax: chex.Array, invalid_actions: Optional[chex.Array]) -> chex.Array:
    """带无效动作掩码的argmax，无效动作设为负无穷"""
    if invalid_actions is not None:
        chex.assert_equal_shape([to_argmax, invalid_actions])
        to_argmax = jnp.where(invalid_actions, -jnp.inf, to_argmax)
    return jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)


def qtransform_by_parent_and_siblings(
    tree: Any,
    node_index: chex.Numeric,
    epsilon: chex.Numeric = 1e-8
) -> chex.Array:
    """MuZero默认Q值转换：归一化到[0,1]区间"""
    qvalues = tree.qvalues(node_index)
    visit_counts = tree.children_visits[node_index]
    node_value = tree.node_values[node_index]
    
    # 未访问动作使用节点价值填充
    safe_qvalues = jnp.where(visit_counts > 0, qvalues, node_value)
    min_val = jnp.minimum(node_value, jnp.min(safe_qvalues))
    max_val = jnp.maximum(node_value, jnp.max(safe_qvalues))
    
    # 归一化计算
    completed_q = jnp.where(visit_counts > 0, qvalues, min_val)
    return (completed_q - min_val) / jnp.maximum(max_val - min_val, epsilon)


def score_considered(
    considered_visit: chex.Numeric,
    gumbel: chex.Array,
    logits: chex.Array,
    normalized_q: chex.Array,
    visit_counts: chex.Array
) -> chex.Array:
    """Gumbel MuZero评分函数：结合Gumbel噪声、先验和Q值"""
    logits = logits - jnp.max(logits, keepdims=True)  # 数值稳定
    # 未达到目标访问次数的动作加惩罚
    penalty = jnp.where(visit_counts == considered_visit, 0, -jnp.inf)
    return gumbel + logits + normalized_q + penalty


def get_table_of_considered_visits(max_considered: int, num_sims: int) -> Tuple:
    """Sequential Halving的访问次数表（Gumbel MuZero根节点用）"""
    def _get_sequence(m: int) -> Tuple:
        if m <= 1:
            return tuple(range(num_sims))
        log2m = int(jnp.ceil(jnp.log2(m)))
        sequence = []
        visits = [0] * m
        num_considered = m
        while len(sequence) < num_sims:
            extra = max(1, num_sims // (log2m * num_considered))
            for _ in range(extra):
                sequence.extend(visits[:num_considered])
                for i in range(num_considered):
                    visits[i] += 1
            num_considered = max(2, num_considered // 2)
        return tuple(sequence[:num_sims])
    
    return tuple(_get_sequence(m) for m in range(max_considered + 1))


# ------------------------------
# 3. 搜索树结构
# ------------------------------

@chex.dataclass(frozen=True)
class Tree:
    """MCTS搜索树结构（支持批量输入）"""
    # 节点属性
    node_visits: chex.Array  # [B, N] 节点访问次数
    node_values: chex.Array  # [B, N] 节点累积价值
    parents: chex.Array  # [B, N] 父节点索引
    action_from_parent: chex.Array  # [B, N] 来自父节点的动作
    embeddings: Any  # [B, N, ...] 节点状态嵌入
    
    # 子节点属性（动作维度）
    children_index: chex.Array  # [B, N, A] 子节点索引
    children_prior_logits: chex.Array  # [B, N, A] 动作先验logits
    children_visits: chex.Array  # [B, N, A] 动作访问次数
    children_rewards: chex.Array  # [B, N, A] 动作奖励
    children_discounts: chex.Array  # [B, N, A] 动作折扣因子
    
    # 根节点属性
    root_invalid_actions: chex.Array  # [B, A] 根节点无效动作掩码
    extra_data: Any  # 额外数据（如Gumbel噪声）
    
    # 常量定义
    ROOT_INDEX: ClassVar[int] = 0
    NO_PARENT: ClassVar[int] = -1
    UNVISITED: ClassVar[int] = -1

    @property
    def num_actions(self) -> int:
        return self.children_index.shape[-1]

    def qvalues(self, indices: chex.Array) -> chex.Array:
        """计算指定节点的Q值：Q(s,a) = r(s,a) + γ*V(s')"""
        if jnp.ndim(indices) == 0:
            return self._unbatched_qvalues(indices)
        return jax.vmap(self._unbatched_qvalues)(indices)

    def _unbatched_qvalues(self, index: int) -> chex.Array:
        return (self.children_rewards[:, index] 
                + self.children_discounts[:, index] * self.node_values[:, self.children_index[:, index]])

    def summary(self) -> SearchSummary:
        """提取根节点搜索结果摘要"""
        batch_size = self.node_values.shape[0]
        root_idx = jnp.full((batch_size,), self.ROOT_INDEX)
        
        visit_counts = self.children_visits[:, self.ROOT_INDEX].astype(jnp.float32)
        total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)
        visit_probs = visit_counts / jnp.maximum(total_counts, 1)  # 避免除零
        
        return SearchSummary(
            visit_counts=visit_counts,
            visit_probs=visit_probs,
            value=self.node_values[:, self.ROOT_INDEX],
            qvalues=self.qvalues(root_idx)
        )


def instantiate_tree(
    root: RootFnOutput,
    num_simulations: int,
    invalid_actions: Optional[chex.Array],
    extra_data: Any
) -> Tree:
    """从根节点初始化搜索树"""
    batch_size, num_actions = root.prior_logits.shape
    num_nodes = num_simulations + 1  # 根节点+模拟次数
    data_dtype = root.value.dtype
    batch_node = (batch_size, num_nodes)
    batch_node_action = (batch_size, num_nodes, num_actions)

    # 初始化空树
    if invalid_actions is None:
        invalid_actions = jnp.zeros((batch_size, num_actions), dtype=jnp.bool_)
    
    # 嵌入初始化（适配任意形状的嵌入）
    def _zero_embedding(x: chex.Array) -> chex.Array:
        return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)
    init_embeddings = jax.tree.map(_zero_embedding, root.embedding)

    tree = Tree(
        # 节点属性
        node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
        node_values=jnp.zeros(batch_node, dtype=data_dtype),
        parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
        action_from_parent=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
        embeddings=init_embeddings,
        
        # 子节点属性
        children_index=jnp.full(batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
        children_prior_logits=jnp.zeros(batch_node_action, dtype=root.prior_logits.dtype),
        children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
        children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
        children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
        
        # 根节点属性
        root_invalid_actions=invalid_actions,
        extra_data=extra_data
    )

    # 填充根节点数据
    root_idx = jnp.full((batch_size,), Tree.ROOT_INDEX)
    tree = tree.replace(
        node_visits=tree.node_visits.at[:, root_idx].set(1),
        node_values=tree.node_values.at[:, root_idx].set(root.value),
        children_prior_logits=tree.children_prior_logits.at[:, root_idx].set(root.prior_logits),
        embeddings=jax.tree.map(
            lambda e, r_e: e.at[:, root_idx].set(r_e),
            tree.embeddings, root.embedding
        )
    )
    return tree


# ------------------------------
# 4. MCTS核心搜索逻辑
# ------------------------------

def simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn: Callable,
    max_depth: int
) -> Tuple[chex.Array, chex.Array]:
    """单条模拟路径：从根节点遍历到叶节点"""
    def cond(state):
        return state["is_continuing"]

    def body(state):
        node_idx = state["next_node_idx"]
        rng_key, sub_key = jax.random.split(state["rng_key"])
        
        # 选择动作
        action = action_selection_fn(sub_key, tree, node_idx, state["depth"])
        next_node_idx = tree.children_index[:, node_idx, action]
        
        # 判断是否继续遍历（未访问节点或未达最大深度）
        depth = state["depth"] + 1
        is_continuing = jnp.logical_and(
            next_node_idx == Tree.UNVISITED,
            depth < max_depth
        )
        
        return {
            "rng_key": rng_key,
            "node_idx": node_idx,
            "action": action,
            "next_node_idx": next_node_idx,
            "depth": depth,
            "is_continuing": is_continuing
        }

    # 初始状态（从根节点开始）
    initial_state = {
        "rng_key": rng_key,
        "node_idx": jnp.full((tree.node_values.shape[0],), Tree.NO_PARENT),
        "action": jnp.full((tree.node_values.shape[0],), Tree.NO_PARENT),
        "next_node_idx": jnp.full((tree.node_values.shape[0],), Tree.ROOT_INDEX),
        "depth": jnp.zeros((tree.node_values.shape[0],), dtype=jnp.int32),
        "is_continuing": jnp.ones((tree.node_values.shape[0],), dtype=jnp.bool_)
    }

    end_state = jax.lax.while_loop(cond, body, initial_state)
    return end_state["node_idx"], end_state["action"]


def expand(
    params: Params,
    rng_key: chex.PRNGKey,
    tree: Tree,
    recurrent_fn: RecurrentFn,
    parent_idx: chex.Array,
    action: chex.Array,
    sim_idx: int
) -> Tree:
    """扩展叶节点：用递归网络预测子节点属性"""
    batch_size = tree.node_values.shape[0]
    batch_range = jnp.arange(batch_size)
    
    # 获取父节点嵌入
    parent_embedding = jax.tree.map(
        lambda e: e[batch_range, parent_idx], tree.embeddings
    )
    
    # 递归网络预测
    rng_keys = jax.random.split(rng_key, batch_size)
    step_out, new_embedding = jax.vmap(recurrent_fn)(
        params, rng_keys, action, parent_embedding
    )
    
    # 新节点索引（第sim_idx次模拟对应第sim_idx+1个节点）
    new_node_idx = jnp.full((batch_size,), sim_idx + 1)
    
    # 更新树结构
    return tree.replace(
        # 新节点属性
        node_visits=tree.node_visits.at[:, new_node_idx].set(1),
        node_values=tree.node_values.at[:, new_node_idx].set(step_out.value),
        parents=tree.parents.at[:, new_node_idx].set(parent_idx),
        action_from_parent=tree.action_from_parent.at[:, new_node_idx].set(action),
        embeddings=jax.tree.map(
            lambda e, ne: e[batch_range, new_node_idx].set(ne),
            tree.embeddings, new_embedding
        ),
        
        # 父节点→新节点的边属性
        children_index=tree.children_index.at[batch_range, parent_idx, action].set(new_node_idx),
        children_prior_logits=tree.children_prior_logits.at[batch_range, parent_idx, action].set(step_out.prior_logits),
        children_rewards=tree.children_rewards.at[batch_range, parent_idx, action].set(step_out.reward),
        children_discounts=tree.children_discounts.at[batch_range, parent_idx, action].set(step_out.discount)
    )


def backward(tree: Tree, leaf_idx: chex.Array) -> Tree:
    """反向更新：从叶节点回溯更新路径上的节点价值"""
    def cond(state):
        _, _, idx = state
        return idx != Tree.ROOT_INDEX

    def body(state):
        tree, leaf_val, idx = state
        parent_idx = tree.parents[:, idx]
        action = tree.action_from_parent[:, idx]
        
        # 计算更新后的价值：V(parent) = (V(parent)*N + (r + γ*V(leaf)))/(N+1)
        parent_visits = tree.node_visits[:, parent_idx]
        reward = tree.children_rewards[:, parent_idx, action]
        discount = tree.children_discounts[:, parent_idx, action]
        updated_val = (tree.node_values[:, parent_idx] * parent_visits 
                       + reward + discount * leaf_val) / (parent_visits + 1)
        
        # 更新父节点属性
        tree = tree.replace(
            node_visits=tree.node_visits.at[:, parent_idx].set(parent_visits + 1),
            node_values=tree.node_values.at[:, parent_idx].set(updated_val),
            children_visits=tree.children_visits.at[:, parent_idx, action].set(
                tree.children_visits[:, parent_idx, action] + 1
            )
        )
        return tree, reward + discount * leaf_val, parent_idx

    # 从叶节点开始回溯
    initial_state = (tree, tree.node_values[:, leaf_idx], leaf_idx)
    tree, _, _ = jax.lax.while_loop(cond, body, initial_state)
    return tree


def mcts_search(
    params: Params,
    rng_key: chex.PRNGKey,
    root: RootFnOutput,
    recurrent_fn: RecurrentFn,
    root_action_fn: Callable,
    interior_action_fn: Callable,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None
) -> Tree:
    """完整MCTS搜索：模拟→扩展→反向更新循环"""
    # 初始化搜索树
    extra_data = None  # 可扩展存储Gumbel噪声等
    tree = instantiate_tree(root, num_simulations, invalid_actions, extra_data)
    max_depth = max_depth or num_simulations

    # 动作选择函数（根节点与内部节点切换）
    def action_selection_fn(rng, tree, node_idx, depth):
        return jax.lax.cond(
            depth == 0,
            lambda x: root_action_fn(*x[:3]),
            lambda x: interior_action_fn(*x),
            (rng, tree, node_idx, depth)
        )

    # 模拟循环
    def simulation_step(i, carry):
        rng_key, tree = carry
        rng_key, sim_key, expand_key = jax.random.split(rng_key, 3)
        
        # 1. 模拟路径
        parent_idx, action = simulate(sim_key, tree, action_selection_fn, max_depth)
        
        # 2. 扩展叶节点（第i次模拟对应第i+1个节点）
        tree = expand(params, expand_key, tree, recurrent_fn, parent_idx, action, i)
        
        # 3. 反向更新
        leaf_idx = jnp.full((tree.node_values.shape[0],), i + 1)
        tree = backward(tree, leaf_idx)
        
        return rng_key, tree

    # 执行num_simulations次模拟
    rng_key, tree = jax.lax.fori_loop(
        0, num_simulations, simulation_step, (rng_key, tree)
    )
    return tree


# ------------------------------
# 5. 动作选择策略
# ------------------------------

def muzero_action_selection(
    rng_key: chex.PRNGKey,
    tree: Tree,
    node_idx: chex.Array,
    depth: chex.Array,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: QTransform = qtransform_by_parent_and_siblings
) -> chex.Array:
    """MuZero PUCT动作选择：结合Q值、先验和访问次数"""
    batch_size = tree.node_values.shape[0]
    node_visits = tree.node_visits[:, node_idx]  # [B]
    children_visits = tree.children_visits[:, node_idx]  # [B, A]
    prior_logits = tree.children_prior_logits[:, node_idx]  # [B, A]
    
    # PUCT公式计算：U(s,a) = C * P(s,a) * sqrt(N(s))/(1+N(s,a))
    pb_c = pb_c_init + jnp.log((node_visits + pb_c_base + 1) / pb_c_base)  # [B]
    prior_probs = jax.nn.softmax(prior_logits)  # [B, A]
    policy_score = pb_c[:, None] * prior_probs * jnp.sqrt(node_visits[:, None]) / (children_visits + 1)
    
    # Q值转换
    q_values = qtransform(tree, node_idx)  # [B, A]
    
    # 加微小噪声打破平局
    noise = 1e-7 * jax.random.uniform(rng_key, prior_probs.shape)
    
    # 根节点应用无效动作掩码
    to_argmax = q_values + policy_score + noise
    invalid_mask = tree.root_invalid_actions * (depth[:, None] == 0)
    
    return masked_argmax(to_argmax, invalid_mask)


@chex.dataclass(frozen=True)
class GumbelExtraData:
    root_gumbel: chex.Array  # [B, A] 根节点Gumbel噪声


def gumbel_muzero_root_action(
    rng_key: chex.PRNGKey,
    tree: Tree,
    node_idx: chex.Array,
    num_simulations: int,
    max_considered: int = 16,
    qtransform: QTransform = qtransform_by_parent_and_siblings
) -> chex.Array:
    """Gumbel MuZero根节点动作选择：Sequential Halving"""
    batch_size = tree.node_values.shape[0]
    children_visits = tree.children_visits[:, node_idx]  # [B, A]
    prior_logits = tree.children_prior_logits[:, node_idx]  # [B, A]
    gumbel = tree.extra_data.root_gumbel  # [B, A]
    
    # 计算当前模拟对应的目标访问次数
    visit_table = get_table_of_considered_visits(max_considered, num_simulations)
    sim_idx = jnp.sum(children_visits, axis=-1).astype(jnp.int32)  # [B]
    num_valid = jnp.sum(1 - tree.root_invalid_actions, axis=-1).astype(jnp.int32)  # [B]
    num_considered = jnp.minimum(max_considered, num_valid)  # [B]
    
    # 从表中获取当前需要考虑的访问次数
    considered_visit = jax.vmap(lambda m, i: visit_table[m][i])(num_considered, sim_idx)  # [B]
    
    # 计算动作评分
    q_values = qtransform(tree, node_idx)  # [B, A]
    to_argmax = score_considered(considered_visit[:, None], gumbel, prior_logits, q_values, children_visits)
    
    return masked_argmax(to_argmax, tree.root_invalid_actions)


def gumbel_muzero_interior_action(
    rng_key: chex.PRNGKey,
    tree: Tree,
    node_idx: chex.Array,
    depth: chex.Array,
    qtransform: QTransform = qtransform_by_parent_and_siblings
) -> chex.Array:
    """Gumbel MuZero内部节点动作选择：确定性选择"""
    del rng_key, depth
    children_visits = tree.children_visits[:, node_idx]  # [B, A]
    prior_logits = tree.children_prior_logits[:, node_idx]  # [B, A]
    q_values = qtransform(tree, node_idx)  # [B, A]
    
    # 改进策略：先验+Q值
    probs = jax.nn.softmax(prior_logits + q_values)  # [B, A]
    sum_visits = jnp.sum(children_visits, axis=-1, keepdims=True)  # [B, 1]
    to_argmax = probs - children_visits / (1 + sum_visits)  # 确定性选择公式
    
    return jnp.argmax(to_argmax, axis=-1).astype(jnp.int32)


# ------------------------------
# 6. 完整策略函数
# ------------------------------

def muzero_policy(
    params: Params,
    rng_key: chex.PRNGKey,
    state: Any,
    root_fn: RootFn,
    recurrent_fn: RecurrentFn,
    num_simulations: int,
    num_actions: int,
    invalid_actions: Optional[chex.Array] = None,
    temperature: float = 1.0
) -> PolicyOutput:
    """MuZero完整策略：表示网络→MCTS→动作决策"""
    # 1. 表示网络生成根节点
    rng_key, root_key = jax.random.split(rng_key)
    root = root_fn(params, root_key, state)
    
    # 2. 定义动作选择函数
    root_action_fn = lambda r, t, n: muzero_action_selection(r, t, n, depth=jnp.zeros_like(n),)
    interior_action_fn = muzero_action_selection
    
    # 3. 执行MCTS搜索
    rng_key, search_key = jax.random.split(rng_key)
    search_tree = mcts_search(
        params=params,
        rng_key=search_key,
        root=root,
        recurrent_fn=recurrent_fn,
        root_action_fn=root_action_fn,
        interior_action_fn=interior_action_fn,
        num_simulations=num_simulations,
        invalid_actions=invalid_actions
    )
    
    # 4. 基于访问次数生成动作
    summary = search_tree.summary()
    action_weights = summary.visit_probs  # [B, A]
    
    # 温度调整（温度=0为贪心选择）
    logits = jnp.log(action_weights + 1e-8) / max(temperature, 1e-8)
    rng_key, action_key = jax.random.split(rng_key)
    action = jax.random.categorical(action_key, logits)  # [B]
    
    return PolicyOutput(
        action=action,
        action_weights=action_weights,
        search_tree=search_tree
    )


def gumbel_muzero_policy(
    params: Params,
    rng_key: chex.PRNGKey,
    state: Any,
    root_fn: RootFn,
    recurrent_fn: RecurrentFn,
    num_simulations: int,
    num_actions: int,
    invalid_actions: Optional[chex.Array] = None,
    max_considered: int = 16,
    gumbel_scale: float = 1.0
) -> PolicyOutput:
    """Gumbel MuZero完整策略：Gumbel噪声+Sequential Halving"""
    # 1. 表示网络生成根节点
    rng_key, root_key, gumbel_key = jax.random.split(rng_key, 3)
    root = root_fn(params, root_key, state)
    batch_size = root.value.shape[0]
    
    # 2. 生成Gumbel噪声
    gumbel = gumbel_scale * jax.random.gumbel(gumbel_key, (batch_size, num_actions))
    extra_data = GumbelExtraData(root_gumbel=gumbel)
    
    # 3. 重新初始化带Gumbel噪声的搜索树
    search_tree = instantiate_tree(
        root=root,
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        extra_data=extra_data
    )
    
    # 4. 定义动作选择函数
    root_action_fn = lambda r, t, n: gumbel_muzero_root_action(
        r, t, n, num_simulations=num_simulations, max_considered=max_considered
    )
    interior_action_fn = gumbel_muzero_interior_action
    
    # 5. 执行MCTS搜索（复用核心循环，替换初始树）
    def simulation_step(i, carry):
        rng_key, tree = carry
        rng_key, sim_key, expand_key = jax.random.split(rng_key, 3)
        
        # 模拟路径
        parent_idx, action = simulate(sim_key, tree, 
            lambda r, t, ni, d: jax.lax.cond(
                d == 0, lambda x: root_action_fn(*x[:3]), lambda x: interior_action_fn(*x), (r, t, ni, d)
            ), max_depth=num_simulations)
        
        # 扩展+反向更新
        tree = expand(params, expand_key, tree, recurrent_fn, parent_idx, action, i)
        leaf_idx = jnp.full((batch_size,), i + 1)
        tree = backward(tree, leaf_idx)
        return rng_key, tree
    
    rng_key, tree = jax.lax.fori_loop(0, num_simulations, simulation_step, (rng_key, search_tree))
    
    # 6. 决策动作（最高Gumbel+先验+Q值）
    summary = tree.summary()
    q_values = qtransform_by_parent_and_siblings(tree, jnp.full((batch_size,), Tree.ROOT_INDEX))
    to_argmax = gumbel + root.prior_logits + q_values
    action = masked_argmax(to_argmax, invalid_actions)
    
    # 生成动作权重（先验+Q值的softmax）
    action_weights = jax.nn.softmax(root.prior_logits + q_values)
    
    return PolicyOutput(
        action=action,
        action_weights=action_weights,
        search_tree=tree
    )


# ------------------------------
# 7. 端到端预训练与推理示例
# ------------------------------

def make_muzero_model(num_actions: int, embedding_dim: int = 32):
    """创建简化的MuZero模型（表示网络+递归网络）"""
    # 表示网络：环境状态→根节点输出
    def root_fn(params: Params, rng_key: chex.PRNGKey, state: chex.Array) -> RootFnOutput:
        # 简化：状态直接线性映射
        prior_logits = jax.nn.Dense(num_actions)(state)
        value = jax.nn.Dense(1)(state).squeeze(-1)
        embedding = jax.nn.Dense(embedding_dim)(state)
        return RootFnOutput(prior_logits=prior_logits, value=value, embedding=embedding)
    
    # 递归网络：(嵌入+动作)→下一状态预测
    def recurrent_fn(params: Params, rng_key: chex.PRNGKey, action: Action, embedding: chex.Array) -> Tuple[RecurrentFnOutput, Any]:
        # 动作嵌入
        action_emb = jax.nn.one_hot(action, num_actions)
        x = jnp.concatenate([embedding, action_emb], axis=-1)
        
        # 预测输出
        reward = jax.nn.Dense(1)(x).squeeze(-1)
        discount = jnp.ones_like(reward) * 0.99  # 固定折扣因子
        prior_logits = jax.nn.Dense(num_actions)(x)
        value = jax.nn.Dense(1)(x).squeeze(-1)
        new_embedding = jax.nn.Dense(embedding_dim)(x)
        
        return RecurrentFnOutput(reward=reward, discount=discount, prior_logits=prior_logits, value=value), new_embedding
    
    # 初始化参数
    rng_key = jax.random.PRNGKey(42)
    dummy_state = jnp.zeros((1, 10))  # 假设环境状态维度为10
    dummy_action = jnp.zeros((1,), dtype=jnp.int32)
    dummy_embedding = jax.nn.Dense(embedding_dim)(dummy_state)
    
    # 实际参数用随机初始化（这里简化为字典结构）
    params = {
        "root": {
            "dense1": jax.random.normal(rng_key, (10, num_actions)),
            "dense2": jax.random.normal(rng_key, (10, 1)),
            "dense3": jax.random.normal(rng_key, (10, embedding_dim))
        },
        "recurrent": {
            "dense1": jax.random.normal(rng_key, (embedding_dim + num_actions, 1)),
            "dense2": jax.random.normal(rng_key, (embedding_dim + num_actions, num_actions)),
            "dense3": jax.random.normal(rng_key, (embedding_dim + num_actions, 1)),
            "dense4": jax.random.normal(rng_key, (embedding_dim + num_actions, embedding_dim))
        }
    }
    return root_fn, recurrent_fn, params


class RandomEnv:
    """随机奖励环境：用于测试训练流程"""
    def __init__(self, state_dim: int = 10, num_actions: int = 4):
        self.state_dim = state_dim
        self.num_actions = num_actions
    
    def reset(self, rng_key: chex.PRNGKey, batch_size: int = 1) -> chex.Array:
        """重置环境，返回初始状态"""
        return jax.random.normal(rng_key, (batch_size, self.state_dim))
    
    def step(self, rng_key: chex.PRNGKey, state: chex.Array, action: Action) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """执行动作，返回(新状态, 奖励, 终止信号)"""
        new_state = jax.random.normal(rng_key, state.shape)
        reward = jax.random.uniform(rng_key, (state.shape[0],))
        done = jnp.zeros((state.shape[0],), dtype=jnp.bool_)
        return new_state, reward, done


def muzero_loss(
    params: Params,
    rng_key: chex.PRNGKey,
    states: chex.Array,
    actions: Action,
    targets: Tuple[chex.Array, chex.Array],
    root_fn: RootFn,
    recurrent_fn: RecurrentFn
) -> chex.Array:
    """MuZero损失函数：策略损失+价值损失"""
    # 目标：(动作权重, 价值)
    target_action_weights, target_values = targets
    
    # 1. 表示网络预测
    root = root_fn(params, rng_key, states)
    policy_loss = jnp.mean(jax.nn.cross_entropy(root.prior_logits, target_action_weights))
    
    # 2. 价值损失
    value_loss = jnp.mean((root.value - target_values) ** 2)
    
    return policy_loss + value_loss


def train_muzero(num_episodes: int = 100, batch_size: int = 2, num_simulations: int = 10):
    # 1. 初始化组件
    num_actions = 4
    env = RandomEnv(num_actions=num_actions)
    root_fn, recurrent_fn, params = make_muzero_model(num_actions=num_actions)
    optimizer = jax.experimental.optimizers.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    
    # 2. 损失函数JIT编译
    loss_fn = jax.jit(muzero_loss)
    grad_fn = jax.jit(jax.grad(loss_fn))
    
    # 3. 训练循环
    rng_key = jax.random.PRNGKey(42)
    for episode in range(num_episodes):
        # 重置环境
        rng_key, reset_key = jax.random.split(rng_key)
        state = env.reset(reset_key, batch_size=batch_size)
        total_reward = 0.0
        
        # 单回合交互
        for step in range(20):  # 最大步数20
            # a. 执行MuZero策略获取动作与搜索结果
            rng_key, policy_key = jax.random.split(rng_key)
            policy_out = muzero_policy(
                params=params,
                rng_key=policy_key,
                state=state,
                root_fn=root_fn,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                num_actions=num_actions,
                temperature=1.0
            )
            
            # b. 环境执行动作
            rng_key, step_key = jax.random.split(rng_key)
            next_state, reward, done = env.step(step_key, state, policy_out.action)
            total_reward += jnp.mean(reward)
            
            # c. 构造训练目标（简化：用搜索权重作为策略目标，环境奖励作为价值目标）
            targets = (policy_out.action_weights, reward)
            
            # d. 计算梯度并更新
            rng_key, loss_key = jax.random.split(rng_key)
            grads = grad_fn(params, loss_key, state, policy_out.action, targets, root_fn, recurrent_fn)
            opt_state = optimizer.update(0, grads, opt_state)
            params = optimizer.get_params(opt_state)
            
            # e. 更新状态
            state = next_state
            if jnp.all(done):
                break
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:3d} | Total Reward: {total_reward:.2f}")
    
    return params, root_fn, recurrent_fn


def infer_muzero(params: Params, root_fn: RootFn, recurrent_fn: RecurrentFn):
    # 初始化环境
    num_actions = 4
    env = RandomEnv(num_actions=num_actions)
    rng_key = jax.random.PRNGKey(123)
    
    # 重置环境
    rng_key, reset_key = jax.random.split(rng_key)
    state = env.reset(reset_key, batch_size=1)
    total_reward = 0.0
    
    # 推理循环（贪心策略，温度=0）
    for step in range(20):
        rng_key, policy_key = jax.random.split(rng_key)
        policy_out = muzero_policy(
            params=params,
            rng_key=policy_key,
            state=state,
            root_fn=root_fn,
            recurrent_fn=recurrent_fn,
            num_simulations=20,  # 推理时可增加模拟次数
            num_actions=num_actions,
            temperature=0.0  # 贪心选择
        )
        
        # 执行动作
        rng_key, step_key = jax.random.split(rng_key)
        state, reward, done = env.step(step_key, state, policy_out.action)
        total_reward += jnp.mean(reward)
        
        print(f"Step {step+1:2d} | Action: {policy_out.action[0]} | Reward: {reward[0]:.2f}")
        if jnp.all(done):
            break
    
    print(f"\nTotal Inference Reward: {total_reward:.2f}")


if __name__ == "__main__":
    # 训练模型
    print("=== Starting Training ===")
    trained_params, root_fn, recurrent_fn = train_muzero(num_episodes=100)
    
    # 推理测试
    print("\n=== Starting Inference ===")
    infer_muzero(trained_params, root_fn, recurrent_fn)
