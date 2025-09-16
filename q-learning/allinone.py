import numpy as np
import random
import matplotlib.pyplot as plt

# 网格环境参数
GRID_SIZE = 5
START = (0, 0)
WIN_STATE = (4, 4)
HOLE_STATES = [(1, 0), (3, 1), (4, 2), (1, 3)]
ACTIONS = [0, 1, 2, 3]  # 上、下、左、右

class QLearner:
    def __init__(self):
        # 初始化参数
        self.alpha = 0.5    # 学习率
        self.gamma = 0.9    # 折扣因子
        self.epsilon = 0.1  # 探索率
        
        # 初始化Q表 (状态-动作值函数)
        self.q_table = {}
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                for a in ACTIONS:
                    self.q_table[(i, j, a)] = 0.0
        
        self.reward_history = []  # 记录每轮奖励

    def get_reward(self, state):
        """获取当前状态的奖励"""
        if state in HOLE_STATES:
            return -5
        if state == WIN_STATE:
            return 1
        return -1  # 每步默认惩罚，鼓励尽快到达目标

    def is_terminal(self, state):
        """判断是否为终止状态"""
        return state == WIN_STATE or state in HOLE_STATES

    def next_state(self, state, action):
        """根据动作计算下一个状态（含边界检查）"""
        i, j = state
        if action == 0:  # 上
            ni, nj = i-1, j
        elif action == 1:  # 下
            ni, nj = i+1, j
        elif action == 2:  # 左
            ni, nj = i, j-1
        else:  # 右
            ni, nj = i, j+1
        
        # 确保不超出网格范围
        if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE:
            return (ni, nj)
        return state  # 撞墙则停在原地

    def choose_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() > self.epsilon:
            #  exploitation: 选择当前状态下Q值最大的动作
            max_q = -float('inf')
            best_action = 0
            for a in ACTIONS:
                if self.q_table[(state[0], state[1], a)] > max_q:
                    max_q = self.q_table[(state[0], state[1], a)]
                    best_action = a
            return best_action
        else:
            # exploration: 随机选择动作
            return random.choice(ACTIONS)

    def train(self, episodes):
        """训练主循环"""
        for _ in range(episodes):
            current_state = START
            total_reward = 0
            
            while not self.is_terminal(current_state):
                # 选择动作并执行
                action = self.choose_action(current_state)
                next_s = self.next_state(current_state, action)
                reward = self.get_reward(next_s)
                
                # 更新Q值: Q(s,a) = Q(s,a) + α[r + γ*maxQ(s',a') - Q(s,a)]
                current_q = self.q_table[(current_state[0], current_state[1], action)]
                max_future_q = max(self.q_table[(next_s[0], next_s[1], a)] for a in ACTIONS)
                new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
                self.q_table[(current_state[0], current_state[1], action)] = round(new_q, 3)
                
                # 移动到下一个状态
                current_state = next_s
                total_reward += reward
            
            self.reward_history.append(total_reward)

    def test(self):
        """测试训练好的模型，输出最优路径"""
        path = [START]
        current_state = START
        
        while not self.is_terminal(current_state):
            action = self.choose_action(current_state)  # 此时主要是 exploitation
            current_state = self.next_state(current_state, action)
            path.append(current_state)
            
            # 防止无限循环（理论上不会发生）
            if len(path) > 100:
                print("路径过长，可能存在问题")
                break
        
        return path

    def plot_training(self):
        """绘制训练过程中的奖励变化"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.reward_history)
        plt.xlabel('训练轮次')
        plt.ylabel('总奖励')
        plt.title('Q-learning训练奖励曲线')
        plt.grid(True)
        plt.show()

    def print_q_table(self):
        """打印每个状态的最大Q值（最优动作价值）"""
        print("网格每个状态的最大Q值:")
        for i in range(GRID_SIZE):
            row = []
            for j in range(GRID_SIZE):
                max_q = max(self.q_table[(i, j, a)] for a in ACTIONS)
                row.append(f"{max_q:.2f}")
            print(" | ".join(row))
            print("-" * (GRID_SIZE * 6 - 1))


# 主程序：端到端流程
if __name__ == "__main__":
    # 1. 创建智能体并训练
    agent = QLearner()
    print("开始训练...")
    agent.train(episodes=5000)  # 训练5000轮
    
    # 2. 可视化训练效果
    agent.plot_training()
    
    # 3. 展示学习到的Q值表
    agent.print_q_table()
    
    # 4. 测试训练好的智能体，输出最优路径
    optimal_path = agent.test()
    print("\n最优路径:", optimal_path)
    print("路径长度:", len(optimal_path))
    print("是否到达终点:", optimal_path[-1] == WIN_STATE)
