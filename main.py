import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 每个节点的参数
nodes = {
    'Node1': {'mean': 50, 'variance': 5},
    'Node2': {'mean': 30, 'variance': 10},
    'Node3': {'mean': 40, 'variance': 7}
}

# 过程参数
thetas = [0.1, 0.5, 0.8]
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 时间步数
time_steps = 1000

# 初始化字典以存储时间序列数据
time_series_data = {node: {} for node in nodes.keys()}

np.random.seed(42)  # 保证结果可复现
for theta in thetas:
    for node, params in nodes.items():
        mean = params['mean']
        variance = params['variance']
        # 初始化时间序列
        x = np.zeros(time_steps)
        x[0] = mean  # 初始值

        for t in range(1, time_steps):
            e_t = np.random.normal(0, np.sqrt(variance))
            x[t] = x[t - 1] + (1 - theta) * (mean - x[t - 1]) + e_t

        time_series_data[node][f'theta_{theta}'] = x

# 转换为DataFrame以显示部分生成的数据
df = pd.DataFrame({(node, theta): data for node, thetas in time_series_data.items() for theta, data in thetas.items()})
print(df.head())


def epsilon_greedy(epsilon, rewards, time_steps):
    num_actions = len(rewards)
    q_values = np.zeros(num_actions)
    action_counts = np.zeros(num_actions)
    regrets = np.zeros(time_steps)
    optimal_reward = np.max([rewards[action][t] for action in range(num_actions) for t in range(time_steps)])

    for t in range(time_steps):
        if np.random.rand() < epsilon:
            action = int(np.random.randint(0, num_actions))
        else:
            action = int(np.argmax(q_values))

        reward = rewards[action][t]
        action_counts[action] += 1
        q_values[action] += (reward - q_values[action]) / action_counts[action]

        regrets[t] = optimal_reward - reward

    return regrets


# 基于生成数据的每个节点的奖励
rewards = {i: data['theta_0.5'] for i, (node, data) in enumerate(time_series_data.items())}

# 比较不同epsilon值的单次遗憾值并记录
results = {}
for epsilon in epsilons:
    regrets = epsilon_greedy(epsilon, rewards, time_steps)
    results[epsilon] = regrets
def smooth_data(data, alpha=0.01):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
    return smoothed

# 绘制单次遗憾值和时间步长T的图像
plt.figure(figsize=(10, 6))
for epsilon, regret in results.items():
    smoothed_regret = smooth_data(regret)
    plt.plot(smoothed_regret, label=f'Epsilon: {epsilon}')
plt.xlabel('Time Steps (T)')
plt.ylabel('Smoothed Instantaneous Regret')
plt.title('Smoothed Instantaneous Regret vs Time Steps for different Epsilon values')
plt.legend()
plt.grid(True)
plt.show()
# 分析theta和epsilon值的影响并绘制图像
for theta in thetas:
    plt.figure(figsize=(10, 6))
    for epsilon in epsilons:
        rewards = {i: data[f'theta_{theta}'] for i, (node, data) in enumerate(time_series_data.items())}
        regrets = epsilon_greedy(epsilon, rewards, time_steps)
        smoothed_regrets = smooth_data(regrets)
        plt.plot(smoothed_regrets, label=f'Epsilon: {epsilon}')
    plt.xlabel('Time Steps (T)')
    plt.ylabel('Smoothed Instantaneous Regret')
    plt.title(f'Smoothed Instantaneous Regret vs Time Steps for Theta: {theta}')
    plt.legend()
    plt.grid(True)
    plt.show()
