import os
import argparse
import math
import random
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from welford import Welford
from dm_control import suite
import matplotlib.pyplot as plt


def init_weights(module, gain):
    """正交权重初始化"""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


# 策略网络（Actor）：输出动作的概率分布
class Actor(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.mu = torch.nn.Linear(64, action_dim)  # 均值
        self.log_sigma = torch.nn.Parameter(torch.zeros(action_dim))  # 标准差（可学习）
        self.dist = torch.distributions.Normal

        # 权重初始化
        self.fc1.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.fc2.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.mu.apply(lambda m: init_weights(m, gain=0.01))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)
        sigma = torch.exp(self.log_sigma)
        return self.dist(mu, sigma)


# 价值网络（Critic）：估计状态价值
class Critic(torch.nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.v = torch.nn.Linear(64, 1)  # 状态价值输出

        # 权重初始化
        self.fc1.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.fc2.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.v.apply(lambda m: init_weights(m, gain=1.0))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.v(x).view(-1)  # 展平为一维向量


def process_obs(time_step):
    """处理DM Control环境的观测（字典转一维数组）"""
    obs_list = []
    for val in time_step.observation.values():
        # 展平所有观测维度并拼接
        obs_list.append(val.flatten() if val.shape else np.array([val]))
    obs = np.concatenate(obs_list)
    return obs, time_step.reward, time_step.last()


class PPO:
    def __init__(self, args):
        self.args = args
        self._init_seed()
        self._init_env()
        self._init_models()

        # 根据模式自动启动训练或评估
        if self.args.mode == "train":
            self._init_train_env()
            self.train()
        elif self.args.mode == "eval":
            self._load_checkpoint()
            self.eval()

    def _init_seed(self):
        """初始化随机种子确保可复现"""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def _init_env(self):
        """初始化环境并计算观测/动作维度"""
        self.env = suite.load(
            domain_name=self.args.domain,
            task_name=self.args.task,
            task_kwargs={'random': self.args.seed}
        )
        # 计算观测维度（所有观测空间的元素总数）
        obs_spec = self.env.observation_spec()
        self.obs_dim = sum(math.prod(spec.shape) for spec in obs_spec.values())
        # 计算动作维度
        action_spec = self.env.action_spec()
        self.action_dim = math.prod(action_spec.shape)
        # 设备配置（CPU/GPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_models(self):
        """初始化Actor和Critic网络"""
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        if self.args.mode == "train":
            self.critic = Critic(self.obs_dim).to(self.device)

    def _init_train_env(self):
        """初始化训练相关组件（日志、优化器、统计器）"""
        # 创建保存目录
        self.exp_dir = f"./log/{self.args.domain}_{self.args.task}/seed_{self.args.seed}"
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.tb_dir = os.path.join(self.exp_dir, "tensorboard")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # 初始化或加载训练状态
        self.obs_stats = Welford()  # 观测归一化统计器
        self.start_ep = 0

        # 优化器
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)

        # 恢复训练
        if self.args.resume:
            self._load_checkpoint(resume=True)

        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=self.tb_dir)

    def _load_checkpoint(self, resume=False):
        """加载模型 checkpoint（训练恢复或评估）"""
        ckpt_path = self.args.checkpoint if not resume else os.path.join(self.model_dir, "backup.ckpt")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.obs_stats = checkpoint['obs_stats']
        
        if resume:
            self.start_ep = checkpoint['episode'] + 1
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_opt.load_state_dict(checkpoint['actor_opt'])
            self.critic_opt.load_state_dict(checkpoint['critic_opt'])
            print(f"恢复训练：从第 {self.start_ep} 回合开始")

    def normalize_obs(self, obs, update=False):
        """观测归一化（训练时更新统计，评估时直接使用）"""
        if update:
            self.obs_stats.add(obs)
        
        # 避免除以零或NaN，裁剪到[-10,10]范围
        mean = self.obs_stats.mean
        var = self.obs_stats.var_s
        if np.isnan(var).any() or (var == 0).any():
            return np.clip(obs - mean, -10, 10)
        return np.clip((obs - mean) / np.sqrt(var), -10, 10)

    def _save_backup(self, episode):
        """保存训练备份（含模型、优化器、统计信息）"""
        checkpoint = {
            'episode': episode,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'obs_stats': self.obs_stats
        }
        torch.save(checkpoint, os.path.join(self.model_dir, "backup.ckpt"))

    def _save_model(self, episode):
        """保存模型快照"""
        torch.save(
            {'actor': self.actor.state_dict(), 'obs_stats': self.obs_stats},
            os.path.join(self.model_dir, f"ep_{episode}.ckpt")
        )

    def _compute_gae(self, rewards, values, next_value):
        """计算广义优势估计（GAE）"""
        gae = 0.0
        advantages = []
        returns = []

        # 从后往前计算
        for t in reversed(range(len(rewards))):
            # 时序差分误差
            delta = rewards[t] + self.args.gamma * next_value - values[t]
            # 累积优势
            gae = delta + self.args.gamma * self.args.gae_lambda * gae
            advantages.insert(0, gae)
            # 回报 = 优势 + 状态价值
            returns.insert(0, gae + values[t])
            next_value = values[t]

        # 优势归一化
        advantages = torch.tensor(advantages, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return torch.tensor(returns, device=self.device), advantages

    def train(self):
        """PPO训练主流程"""
        print(f"开始训练：设备={self.device}, 总回合数={self.args.episodes}")
        total_steps = 0  # 累计环境步数

        # 数据缓存（收集满update_every步后更新）
        obs_buf, act_buf, log_prob_buf, value_buf, reward_buf = [], [], [], [], []

        for episode in range(self.start_ep, self.args.episodes):
            ep_reward = 0.0
            # 重置环境并处理初始观测
            time_step = self.env.reset()
            obs, _, done = process_obs(time_step)
            obs = self.normalize_obs(obs, update=True)

            while not done:
                # 1. 收集数据
                obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    dist = self.actor(obs_tensor)
                    value = self.critic(obs_tensor)

                # 采样动作并记录数据
                action = dist.sample()
                obs_buf.append(obs_tensor[0])
                act_buf.append(action[0])
                log_prob_buf.append(dist.log_prob(action).sum())
                value_buf.append(value[0])

                # 执行动作
                time_step = self.env.step(action.cpu().numpy()[0])
                next_obs, reward, done = process_obs(time_step)
                next_obs = self.normalize_obs(next_obs, update=True)

                # 更新统计
                ep_reward += reward
                reward_buf.append(reward)
                total_steps += 1
                obs = next_obs

                # 2. 满update_every步或回合结束时更新模型
                if total_steps % self.args.update_every == 0:
                    # 计算最后一个状态的价值
                    with torch.no_grad():
                        next_value = self.critic(torch.tensor(obs, device=self.device).unsqueeze(0))[0]

                    # 计算GAE优势和回报
                    returns, advantages = self._compute_gae(reward_buf, value_buf, next_value)

                    # 转换为张量
                    obs_tensor = torch.stack(obs_buf)
                    act_tensor = torch.stack(act_buf)
                    old_log_probs = torch.stack(log_prob_buf)

                    # 3. PPO多轮更新
                    for _ in range(self.args.epochs):
                        # 随机打乱数据（mini-batch）
                        indices = torch.randperm(len(obs_tensor))
                        for i in range(0, len(obs_tensor), self.args.batch_size):
                            batch_idx = indices[i:i+self.args.batch_size]

                            # 计算当前策略的分布和价值
                            dist = self.actor(obs_tensor[batch_idx])
                            current_values = self.critic(obs_tensor[batch_idx])
                            current_log_probs = dist.log_prob(act_tensor[batch_idx]).sum(1)

                            # 计算重要性权重
                            ratio = torch.exp(current_log_probs - old_log_probs[batch_idx].clamp(min=torch.finfo(torch.float64).min))

                            # 4. 计算损失
                            # Actor损失（PPO裁剪目标）
                            surr1 = ratio * advantages[batch_idx]
                            surr2 = ratio.clamp(1-self.args.clip, 1+self.args.clip) * advantages[batch_idx]
                            actor_loss = -(torch.min(surr1, surr2) + self.args.entropy_coef * dist.entropy().sum(1)).mean()

                            # Critic损失（均方误差）
                            critic_loss = 0.5 * F.mse_loss(current_values, returns[batch_idx])

                            # 5. 反向传播更新
                            self.critic_opt.zero_grad()
                            critic_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_clip)
                            self.critic_opt.step()

                            self.actor_opt.zero_grad()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip)
                            self.actor_opt.step()

                    # 清空缓存
                    obs_buf, act_buf, log_prob_buf, value_buf, reward_buf = [], [], [], [], []

            # 回合结束处理
            self.writer.add_scalar("train/episode_reward", ep_reward, episode)
            print(f"回合 {episode:4d} | 奖励: {ep_reward:.2f}")

            # 定期评估和保存
            if (episode + 1) % self.args.eval_every == 0 or episode == self.args.episodes - 1:
                eval_reward = self.eval(render=False, save_video=False)
                self.writer.add_scalar("eval/average_reward", eval_reward, episode)
                self._save_model(episode)

            # 定期保存备份
            if (episode + 1) % 250 == 0 or episode == self.args.episodes - 1:
                self._save_backup(episode)

        self.writer.close()
        print("训练结束！")

    def eval(self, render=False, save_video=False):
        """模型评估（计算平均奖励，可选渲染/存视频）"""
        print("开始评估...")
        ep_rewards = []
        video_dir = f"./media/{self.args.domain}_{self.args.task}"
        os.makedirs(video_dir, exist_ok=True)
        frame_idx = 0

        for episode in range(self.args.eval_episodes):
            ep_reward = 0.0
            time_step = self.env.reset()
            obs, _, done = process_obs(time_step)
            obs = self.normalize_obs(obs)  # 评估时不更新统计

            while not done:
                # 确定性策略（取均值而非采样）
                obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    dist = self.actor(obs_tensor)
                    action = dist.mean  # 评估用均值更稳定

                # 执行动作
                time_step = self.env.step(action.cpu().numpy()[0])
                next_obs, reward, done = process_obs(time_step)

                # 更新统计
                ep_reward += reward
                obs = self.normalize_obs(next_obs)

                # 渲染和存视频
                if render or save_video:
                    img = self.env.physics.render(height=240, width=240, camera_id=0)
                    if render:
                        plt.imshow(img)
                        plt.axis('off')
                        plt.pause(0.01)
                        plt.draw()
                    if save_video:
                        plt.imsave(os.path.join(video_dir, f"frame_{frame_idx:04d}.png"), img)
                        frame_idx += 1

            ep_rewards.append(ep_reward)
            if render:
                print(f"评估回合 {episode} | 奖励: {ep_reward:.2f}")
                plt.pause(0.5)

        avg_reward = np.mean(ep_rewards)
        print(f"评估完成 | 平均奖励: {avg_reward:.2f}")

        # 视频转MP4（需安装ffmpeg）
        if save_video and frame_idx > 0:
            os.system(
                f"ffmpeg -y -i {video_dir}/frame_%04d.png -r 10 -vf pad=ceil(iw/2)*2:ceil(ih/2)*2 -pix_fmt yuv420p {video_dir}/result.mp4"
            )
            # 删除中间图片
            for img_file in os.listdir(video_dir):
                if img_file.endswith(".png"):
                    os.remove(os.path.join(video_dir, img_file))
            print(f"视频已保存至: {video_dir}/result.mp4")

        return avg_reward


def parse_args():
    """参数解析（默认值适配cartpole-swingup任务，可直接运行）"""
    parser = argparse.ArgumentParser(description="简化版PPO算法")
    # 环境配置
    parser.add_argument("--domain", type=str, default="cartpole", help="环境领域（如cartpole/reacher）")
    parser.add_argument("--task", type=str, default="swingup", help="环境任务（如swingup/hard）")
    parser.add_argument("--mode", type=str, default="train", help="运行模式（train/eval）")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")

    # 训练配置
    parser.add_argument("--episodes", type=int, default=1000, help="训练总回合数")
    parser.add_argument("--resume", action="store_true", help="恢复之前的训练")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE参数lambda")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="熵奖励系数")
    parser.add_argument("--update_every", type=int, default=2048, help="每多少步更新一次模型")
    parser.add_argument("--epochs", type=int, default=10, help="每次更新的训练轮数")
    parser.add_argument("--clip", type=float, default=0.2, help="PPO裁剪系数")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch大小")
    parser.add_argument("--grad_clip", type=float, default=0.5, help="梯度裁剪阈值")

    # 评估配置
    parser.add_argument("--checkpoint", type=str, default="", help="评估用的模型路径")
    parser.add_argument("--eval_episodes", type=int, default=10, help="评估回合数")
    parser.add_argument("--eval_every", type=int, default=50, help="训练时每多少回合评估一次")
    parser.add_argument("--render", action="store_true", help="评估时渲染画面")
    parser.add_argument("--save_video", action="store_true", help="评估时保存视频")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 评估模式必须指定checkpoint
    if args.mode == "eval" and not args.checkpoint:
        raise ValueError("评估模式必须通过 --checkpoint 指定模型路径！")
    PPO(args)