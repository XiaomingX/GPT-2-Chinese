import numpy as np
import random
import time
import os

# 迷宫环境类
class Maze:
    def __init__(self, size=10):
        self.size = size
        self.maze = self._generate()  # 生成迷宫
        self.start = (0, 0)
        self.end = (size-1, size-1)
        # 确保起点和终点无障碍物
        self.maze[self.start] = 0
        self.maze[self.end] = 0

    def _generate(self):
        """生成随机迷宫：0为通路，1为障碍物（约30%）"""
        maze = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < 0.3:
                    maze[i][j] = 1
        return maze

    def is_valid(self, x, y):
        """判断(x,y)是否为有效位置（在边界内且不是障碍物）"""
        return 0 <= x < self.size and 0 <= y < self.size and self.maze[x][y] == 0

    def print(self, a_pos, b_pos):
        """打印迷宫，显示两个智能体位置"""
        os.system('clear')  # 清屏
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if (i, j) == self.start:
                    row.append('S')
                elif (i, j) == self.end:
                    row.append('G')
                elif (i, j) == a_pos:
                    row.append('A')  # Q学习智能体
                elif (i, j) == b_pos:
                    row.append('B')  # 随机智能体
                elif self.maze[i][j] == 1:
                    row.append('#')
                else:
                    row.append(' ')
            print('|'.join(row))
            print('-' * (self.size * 2 - 1))


# Q学习智能体
class QAgent:
    def __init__(self, maze, epsilon=0.9, alpha=0.1, gamma=0.9):
        self.maze = maze
        self.epsilon = epsilon  # 探索率（90%选最优，10%随机）
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子（未来奖励的权重）
        self.actions = [(-1,0), (1,0), (0,-1), (0,1)]  # 上、下、左、右
        # Q表：[x][y][动作索引] = 价值
        self.q_table = np.zeros((maze.size, maze.size, 4))

    def choose_action(self, state, train=True):
        """选择动作：训练时用ε-贪婪，测试时用最优"""
        x, y = state
        # 筛选有效动作（不撞墙、不出界）
        valid_idx = []
        for i, (dx, dy) in enumerate(self.actions):
            if self.maze.is_valid(x+dx, y+dy):
                valid_idx.append(i)
        if not valid_idx:
            return None  # 死胡同
        
        if train and random.random() < self.epsilon:
            return random.choice(valid_idx)  # 探索：随机选有效动作
        else:
            # 利用：选Q值最大的有效动作
            max_q = max(self.q_table[x, y, idx] for idx in valid_idx)
            best_idx = [i for i in valid_idx if self.q_table[x, y, i] == max_q]
            return random.choice(best_idx)

    def learn(self, state, action_idx, reward, next_state):
        """更新Q表：Q(s,a) = Q(s,a) + α[r + γ*maxQ(s',a') - Q(s,a)]"""
        x, y = state
        nx, ny = next_state
        # 当前Q值
        current_q = self.q_table[x, y, action_idx]
        # 下一状态的最大Q值
        max_next_q = np.max(self.q_table[nx, ny, :])
        # 更新公式
        self.q_table[x, y, action_idx] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def train(self, episodes=1000):
        """训练智能体：在迷宫中试错学习"""
        print(f"开始训练（{episodes}回合）...")
        for ep in range(episodes):
            state = self.maze.start
            steps = 0
            while state != self.maze.end:
                action_idx = self.choose_action(state)
                if action_idx is None:
                    break  # 死胡同，重新开始
                
                # 执行动作
                x, y = state
                dx, dy = self.actions[action_idx]
                next_state = (x+dx, y+dy)
                steps += 1

                # 奖励设置：到达终点+100，否则-1（鼓励少走步）
                reward = 100 if next_state == self.maze.end else -1
                # 学习更新
                self.learn(state, action_idx, reward, next_state)
                state = next_state

            # 每100回合打印进度
            if (ep+1) % 100 == 0:
                print(f"回合 {ep+1}/{episodes}，步数：{steps}")


# 随机智能体（用于对比）
class RandomAgent:
    def __init__(self, maze):
        self.maze = maze
        self.actions = [(-1,0), (1,0), (0,-1), (0,1)]  # 上下左右

    def choose_action(self, state):
        """随机选择有效动作"""
        x, y = state
        valid_acts = []
        for dx, dy in self.actions:
            if self.maze.is_valid(x+dx, y+dy):
                valid_acts.append((dx, dy))
        return random.choice(valid_acts) if valid_acts else None


# 模拟对比函数
def simulate(maze, q_agent, random_agent):
    """对比Q学习智能体和随机智能体的表现"""
    print("\n开始模拟（A是Q学习智能体，B是随机智能体）...")
    print("S=起点，G=终点，#=障碍物 | 按Ctrl+C结束")
    time.sleep(2)

    a_pos, b_pos = maze.start, maze.start  # 初始位置
    a_steps, b_steps = 0, 0
    a_done, b_done = False, False

    while not (a_done and b_done):
        # 打印当前状态
        maze.print(a_pos, b_pos)
        print(f"步数：Q智能体={a_steps}，随机智能体={b_steps}")

        # Q智能体移动（测试模式：不探索）
        if not a_done:
            act_idx = q_agent.choose_action(a_pos, train=False)
            if act_idx:
                dx, dy = q_agent.actions[act_idx]
                a_pos = (a_pos[0]+dx, a_pos[1]+dy)
                a_steps += 1
                a_done = (a_pos == maze.end)

        # 随机智能体移动
        if not b_done:
            act = random_agent.choose_action(b_pos)
            if act:
                b_pos = (b_pos[0]+act[0], b_pos[1]+act[1])
                b_steps += 1
                b_done = (b_pos == maze.end)

        time.sleep(0.5)  # 延迟，方便观察

    # 结束状态
    maze.print(a_pos, b_pos)
    print(f"\n模拟结束！Q智能体用了{a_steps}步，随机智能体用了{b_steps}步")


# 主程序（端到端流程）
if __name__ == "__main__":
    # 1. 创建迷宫环境
    maze = Maze(size=10)
    print("初始迷宫：")
    maze.print(maze.start, maze.start)
    time.sleep(2)

    # 2. 训练Q学习智能体
    q_agent = QAgent(maze)
    q_agent.train(episodes=1000)  # 预训练1000回合

    # 3. 创建随机智能体（对比用）
    random_agent = RandomAgent(maze)

    # 4. 模拟对比两个智能体
    simulate(maze, q_agent, random_agent)
