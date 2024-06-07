import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

from tf_agents.environments import gym_wrapper
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common

from traffic_env import TrafficEnv

# 将自定义环境包装为TF-Agents环境
gym_env = gym_wrapper.GymWrapper(TrafficEnv())
tf_env = tf_py_environment.TFPyEnvironment(gym_env)

# 定义Q网络
q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=(100,)
)

# 定义DQN智能体
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.compat.v2.Variable(0)

agent = dqn_agent.DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter
)

agent.initialize()

# 定义策略
eval_policy = agent.policy
collect_policy = agent.collect_policy

# 定义重放缓冲区
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=10000
)

# 定义数据收集和训练步骤
def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # 将轨迹写入重放缓冲区
    buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

# 训练智能体
num_iterations = 10000
batch_size = 64
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2
).prefetch(3)

iterator = iter(dataset)
agent.train = common.function(agent.train)

# 初始化
agent.train_step_counter.assign(0)
collect_data(tf_env, random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec()), replay_buffer, steps=100)

# 训练循环
for _ in range(num_iterations):
    collect_data(tf_env, collect_policy, replay_buffer, steps=1)
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

# 测试智能体
num_episodes = 10
for _ in range(num_episodes):
    time_step = tf_env.reset()
    while not time_step.is_last():
        action_step = eval_policy.action(time_step)
        time_step = tf_env.step(action_step.action)

# 可视化训练结果
rewards = []
obs = gym_env.reset()
for _ in range(100):
    action = eval_policy.action(obs)
    obs, reward, done, _ = gym_env.step(action.numpy())
    rewards.append(reward)
    if done:
        obs = gym_env.reset()

plt.plot(rewards)
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('RL Model Performance in Traffic Environment')
plt.show()
