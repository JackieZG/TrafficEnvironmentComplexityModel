import gym
from gym import spaces
import numpy as np


class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()
        # 状态空间包括11个静态影响因素
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        # 动作空间包括3种动作：直行、左转、右转
        self.action_space = spaces.Discrete(3)
        self.state = None

    def reset(self):
        # 初始化状态
        self.state = np.random.rand(11)
        return self.state

    def step(self, action):
        # 简化的状态转换
        self.state = np.random.rand(11)
        static_complexity = self.calculate_static_complexity(self.state)
        dynamic_complexity = self.calculate_dynamic_complexity()
        total_complexity = static_complexity + dynamic_complexity

        # 根据动作计算奖励
        if action == 0:  # 直行
            reward = -total_complexity
        elif action == 1:  # 左转
            reward = -total_complexity * 1.1
        else:  # 右转
            reward = -total_complexity * 0.9

        done = False  # 简化处理，环境永不结束
        return self.state, reward, done, {}

    def calculate_static_complexity(self, state):
        # 静态复杂度计算示例
        entropy = -np.sum(state * np.log2(state + 1e-9))
        return entropy

    def calculate_dynamic_complexity(self):
        # 动态复杂度计算示例
        G = 1e-6
        R = 1
        Mp = 1500
        Mq = 2000
        S = np.linalg.norm(self.state)
        return abs(G * R * Mp * Mq / (S + 1e-9))


# 测试自定义环境
if __name__ == "__main__":
    env = TrafficEnv()
    obs = env.reset()
    print("Initial observation:", obs)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Next observation:", obs)
    print("Reward:", reward)
