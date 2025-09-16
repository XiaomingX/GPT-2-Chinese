# 核心库导入（去重后）
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          GenerationConfig, load_dataset)
import torch
import torch.distributed as dist
import deepspeed
import requests
import random
import time
import re
from tqdm import tqdm
import os

# -------------------------- 1. 配置参数（集中管理，便于修改） --------------------------
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# 模型与数据配置
model_path = "/data2/Qwen/Qwen2.5-7B"  # 预训练模型路径
dataset_name = "openai/gsm8k"          # 训练数据集（数学推理）
save_dir = "./grpo_trained_models"     # 模型保存目录
os.makedirs(save_dir, exist_ok=True)

# 训练超参数
Q_batch_size = 1  # 每次生成的问题数量（原代码强制为1，保留断言）
num_pre_Q = 8     # 每个问题生成的候选答案数量
all_steps = 1000  # 总训练步数
save_steps = 200  # 每多少步保存一次模型
max_prompt_length = 400  # 最大提示长度

# GRPO算法参数
beta = 0.04       # KL散度权重
clip_param = 0.2  # PPO裁剪参数
compute_gen_logps = True  # 是否计算生成logps（用于裁剪）

# 服务端配置
ref_server = "http://localhost:59875"  # 参考模型服务端地址

# DeepSpeed配置（分布式训练）
ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size * num_pre_Q,
    "gradient_accumulation_steps": 2,
    "optimizer": {"type": "AdamW", "params": {"lr": 1e-6}},
    "bf16": {"enabled": True},  # 混合精度训练
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "offload_optimizer": {"device": "cpu"}  # 优化器卸载到CPU节省显存
    }
}

assert Q_batch_size == 1, "原代码逻辑依赖batch_size=1，暂不支持修改"

# -------------------------- 2. 分布式初始化（必需，否则报错） --------------------------
def init_distributed():
    dist.init_process_group(backend='nccl')  # 多GPU通信后端
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

# -------------------------- 3. 数据加载与预处理 --------------------------
def load_and_process_data():
    """加载gsm8k数据集，处理为(Q, A)格式"""
    dataset = load_dataset(dataset_name, "main", split="train")
    # 提取问题和答案（答案取####后的数值）
    QAs = [
        {'Q': item['question'], 'A': item['answer'].split('####')[-1].strip()}
        for item in dataset
    ]
    return QAs

# -------------------------- 4. 模型与分词器初始化 --------------------------
def init_model_tokenizer(local_rank):
    """初始化分词器和模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 设置pad_token（部分模型默认无pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 初始化模型（交给DeepSpeed管理设备）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa",  # 高效注意力实现
        device_map={"": local_rank}    # 绑定到当前GPU
    )
    return tokenizer, model

# -------------------------- 5. 核心功能函数 --------------------------
def gen_answers(prompts, tokenizer, gen_model, generation_config):
    """根据问题生成候选答案"""
    # 构造对话模板
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within   and <|FunctionCallEnd|> <|FunctionCallEnd|> tags, respectively, i.e., <RichMediaReference> reasoning process here </RichMediaReference> answer here <|FunctionCallEnd|>."""
    
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True
        ) for q in prompts
    ]

    # tokenize输入
    inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False
    )
    prompt_length = inputs["input_ids"].shape[-1]
    
    # 跳过过长的提示
    if prompt_length > max_prompt_length:
        return []
    
    # 生成答案
    inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = gen_model.generate(**inputs, generation_config=generation_config)
    
    # 提取生成部分（去掉提示）并解码
    completion_ids = outputs[:, prompt_length:]
    answers = [tokenizer.decode(ids).replace('<|endoftext|>', '') for ids in completion_ids]
    return answers

def compute_rewards(qa_item, answers):
    """计算答案的奖励（正确性+格式合规性）"""
    rewards = []
    for ans in answers:
        # 1. 正确性奖励（-1或1）
        correct_reward = reward_correct(qa_item, ans)
        # 2. 格式奖励（-1或1.25）
        format_reward = reward_format(ans)
        rewards.append(correct_reward + format_reward)
    return torch.tensor(rewards, dtype=torch.float32)

def reward_correct(qa_item, answer):
    """判断答案正确性（提取最后一个数字与真值对比）"""
    # 导入数学验证工具（需提前安装或实现math_verify模块）
    from math_verify import parse, verify, ExprExtractionConfig
    
    # 提取答案中的数字（支持整数、小数、分数）
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if not nums:
        return -1.0  # 无数字则奖励-1
    
    # 解析最后一个数字与真值
    last_num = nums[-1]
    pred = parse(last_num, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(qa_item["A"], extraction_config=[ExprExtractionConfig()])
    return 1.0 if verify(pred, ground_truth) else -1.0

def reward_format(answer):
    """判断答案格式是否符合要求（是否包含指定标签）"""
    pattern = r"^.*?\s*.*?<|FunctionCallEnd|>$"  # 匹配 推理标签+答案标签
    return 1.25 if re.match(pattern, answer, re.DOTALL) else -1.0

def gen_samples(qa_items, tokenizer, gen_model, generation_config):
    """生成训练样本（提示+答案+奖励）"""
    prompts = [item["Q"] for item in qa_items]
    # 生成候选答案
    answers = gen_answers(prompts, tokenizer, gen_model, generation_config)
    if not answers:
        return None, None, None
    
    # 计算奖励
    rewards = compute_rewards(qa_items[0], answers)  # 因Q_batch_size=1，取第一个问题
    
    # Tokenize提示和答案
    # 提示的tokenize（带对话模板）
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within   and <|FunctionCallEnd|> <|FunctionCallEnd|> tags, respectively, i.e.,  reasoning process here <|FunctionCallEnd|> answer here <|FunctionCallEnd|>."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": prompts[0]}],
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    
    # 答案的tokenize
    answer_ids = tokenizer(
        answers,
        return_tensors="pt",
        padding=True,
        padding_side="right",
        add_special_tokens=False
    )["input_ids"]
    
    return prompt_ids, answer_ids, rewards

def get_per_token_logps(logits, input_ids):
    """计算每个token的对数概率"""
    per_token_logps = []
    for logit_row, id_row in zip(logits, input_ids):
        log_probs = logit_row.log_softmax(dim=-1)  # 转为对数概率分布
        # 取出每个输入token对应的概率
        token_logp = torch.gather(log_probs, 1, id_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_logp)
    return torch.stack(per_token_logps)

def GRPO_step(batch, engine, tokenizer):
    """GRPO损失计算步骤（核心训练逻辑）"""
    # 解析batch数据
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)  # (batch_size, seq_len)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)  # 优势（奖励）
    ref_logps = batch['refs'].to(engine.device)  # 参考模型的logps
    
    # 模型前向计算
    logits = engine(inputs).logits  # (batch_size, seq_len, vocab_size)
    logits = logits[:, :-1, :]      # 去掉最后一个logit（无对应输入）
    input_ids = inputs[:, 1:]       # 去掉第一个输入（无对应logit）
    
    # 计算当前模型的token logps（只保留生成部分）
    curr_logps = get_per_token_logps(logits, input_ids)
    curr_logps = curr_logps[:, prompt_length-1:]
    
    # 1. 计算KL散度惩罚（与参考模型的差异）
    per_token_kl = torch.exp(ref_logps - curr_logps) - (ref_logps - curr_logps) - 1
    
    # 2. 计算生成部分的掩码（排除pad token）
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    # 3. 计算策略损失（PPO裁剪或普通策略梯度）
    if compute_gen_logps:
        # PPO风格：裁剪比率避免更新过大
        ratio = torch.exp(curr_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        # 普通策略梯度
        per_token_loss = torch.exp(curr_logps - curr_logps.detach()) * advantages
    
    # 4. 总损失 = 负的（优势*比率 - KL惩罚）
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    # 按掩码平均（只计算有效token）
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss

def generate_mode(num_samples, rank, QAs, tokenizer, gen_model, generation_config):
    """生成样本并上传到服务端"""
    if rank == 0:
        print(f"开始生成{num_samples}个样本...")
    
    for _ in range(num_samples):
        # 随机采样问题
        inputs = random.sample(QAs, Q_batch_size)
        # 生成样本
        prompt_ids, answer_ids, rewards = gen_samples(inputs, tokenizer, gen_model, generation_config)
        if prompt_ids is None:
            continue
        
        # 奖励标准化（便于训练）
        if (rewards.max() - rewards.min()).item() < 0.01:
            continue  # 奖励差异过小，跳过
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        
        # 合并提示和答案的token ids
        prompt_len = prompt_ids.shape[1]
        # 重复提示以匹配答案数量（每个问题生成num_pre_Q个答案）
        repeated_prompts = prompt_ids.repeat(num_pre_Q, 1)
        merged_ids = torch.cat([repeated_prompts, answer_ids], dim=1)
        
        # 打包数据
        data = [
            json.dumps({"plen": prompt_len}).encode(),  # 提示长度
            tensor_to_bytes(merged_ids),               # 合并的token ids
            tensor_to_bytes(rewards)                   # 奖励
        ]
        
        # 计算生成时的logps（如果需要）
        if compute_gen_logps:
            with torch.inference_mode():
                logits = gen_model(merged_ids.to(gen_model.device)).logits[:, :-1, :]
                gen_logps = get_per_token_logps(logits, merged_ids[:, 1:].to(gen_model.device))
            gen_logps = gen_logps[:, prompt_len-1:].cpu()
            data.append(tensor_to_bytes(gen_logps))
        
        # 上传到服务端
        try:
            requests.post(f"{ref_server}/upload", data=make_bytes_list(data))
        except Exception as e:
            if rank == 0:
                print(f"上传失败: {e}")
            continue

# 工具函数（从服务端脚本复用，避免重复导入）
def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def make_bytes_list(blist: list[bytes]) -> bytes:
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()

def get_batch_from_server():
    """从服务端获取带参考logps的训练batch"""
    try:
        response = requests.get(f"{ref_server}/get").content
        if response == b'empty':
            return None
        # 解析数据
        data_list = bytes_list_to_list(response)
        batch = {
            'plen': json.loads(data_list[0])['plen'],
            'inputs': bytes_to_tensor(data_list[1]),
            'rewards': bytes_to_tensor(data_list[2]),
            'refs': bytes_to_tensor(data_list[3])
        }
        if len(data_list) == 5:
            batch['gen_logps'] = bytes_to_tensor(data_list[4])
        return batch
    except Exception as e:
        print(f"获取batch失败: {e}")
        return None

def bytes_to_tensor(b: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(b), weights_only=True)

def bytes_list_to_list(b: bytes) -> list[bytes]:
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

# -------------------------- 6. 训练主流程 --------------------------
def main():
    # 1. 分布式初始化
    local_rank = init_distributed()
    rank = dist.get_rank()
    
    # 2. 加载数据
    QAs = load_and_process_data()
    if rank == 0:
        print(f"加载数据集完成，共{len(QAs)}个问题")
    
    # 3. 初始化模型和分词器
    tokenizer, model = init_model_tokenizer(local_rank)
    
    # 4. 生成配置
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.9,
        num_return_sequences=num_pre_Q,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 5. 仅生成模式（可选）
    if 'genonly' in sys.argv:
        generate_mode(999999, rank, QAs, tokenizer, model, generation_config)
        return
    
    # 6. 初始化DeepSpeed引擎（分布式训练核心）
    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=model.parameters()
    )
    gen_model = engine  # 生成样本时使用训练引擎（支持分布式）
    
    # 7. 预生成一批样本
    generate_mode(num_samples=10, rank=rank, QAs=QAs, tokenizer=tokenizer,
                  gen_model=gen_model, generation_config=generation_config)
    
    # 8. 开始训练
    progress = range(1, all_steps + 1)
    if rank == 0:
        progress = tqdm(progress, desc="Training")
    
    for step in progress:
        # 获取训练batch（若为空则生成样本）
        batch = get_batch_from_server()
        while batch is None:
            generate_mode(num_samples=2, rank=rank, QAs=QAs, tokenizer=tokenizer,
                          gen_model=gen_model, generation_config=generation_config)
            batch = get_batch_from_server()
        
        # 计算损失
        loss = GRPO_step(batch, engine, tokenizer)
        
        # 反向传播和参数更新
        engine.backward(loss)
        engine.step()
        
        # 更新进度条
        if rank == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")
        
        # 保存模型
        if step % save_steps == 0:
            dist.barrier()  # 等待所有进程同步
            if rank == 0:
                print(f"\n保存模型到 {save_dir}/step_{step}")
                # 保存模型和分词器
                save_path = f"{save_dir}/step_{step}"
                engine.module.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
            dist.barrier()
    
    # 训练结束
    if rank == 0:
        print("训练完成！")

# -------------------------- 7. 训练后模型使用示例 --------------------------
def infer_with_trained_model(trained_model_path, question):
    """使用训练后的模型进行推理"""
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        trained_model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa"
    ).to('cuda')
    model.eval()
    
    # 构造对话提示
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within   and  <|FunctionCallEnd|> tags, respectively, i.e.,  reasoning process here <|FunctionCallEnd|> answer here <|FunctionCallEnd|>."""
    
    chat_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 推理
    inputs = tokenizer(chat_text, return_tensors="pt").to('cuda')
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
    
    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=generation_config)
    
    # 解析结果
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取生成部分（去掉提示）
    generated_part = result.split(tokenizer.apply_chat_template([], add_generation_prompt=True))[-1]
    return generated_part

if __name__ == '__main__':
    import sys
    import io

    # 启动训练
    main()
    
    # 训练完成后运行推理示例（仅主进程运行）
    if dist.get_rank() == 0:
        print("\n=== 推理示例 ===")
        # 选择最后保存的模型
        trained_model_paths = sorted([p for p in os.listdir(save_dir) if p.startswith('step_')])
        if trained_model_paths:
            latest_model = f"{save_dir}/{trained_model_paths[-1]}"
            question = "A bakery makes 120 donuts per hour. They operate for 8 hours a day, 6 days a week. How many donuts do they make in 4 weeks?"
            print(f"问题: {question}")
            print(f"模型回答: {infer_with_trained_model(latest_model, question)}")