import pandas as pd
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

file_path = os.path.expanduser("~/Desktop/MAB_Exp/RasberryPi.csv")

# 加载数据集
vm_df = pd.read_csv(file_path)

# 检查数据集的列名和前几行数据
print(vm_df.columns)
print(vm_df.head())

# 确保 'Execution Time' 列中的所有值都是数值类型
vm_df['Execution Time'] = pd.to_numeric(vm_df['Execution Time'], errors='coerce')

# 删除缺失值
vm_df = vm_df.dropna(subset=['Execution Time'])

# 计算所有execution time的均值和方差
mean_execution_time = vm_df['Execution Time'].mean()
var_execution_time = vm_df['Execution Time'].var()

print(f"Mean Execution Time: {mean_execution_time}")
print(f"Variance Execution Time: {var_execution_time}")

# 定义epsilon-greedy算法
def epsilon_greedy(rewards, T, epsilon):
    num_actions = len(rewards)
    q_values = np.zeros(num_actions)
    action_counts = np.zeros(num_actions)
    regrets = np.zeros(T)
    optimal_rewards = np.max([rewards[action] for action in range(num_actions)], axis=0)

    for t in range(T):
        if np.random.rand() < epsilon:
            action = int(np.random.randint(0, num_actions))
        else:
            action = int(np.argmax(q_values))

        reward = rewards[action][t]
        action_counts[action] += 1
        q_values[action] += (reward - q_values[action]) / action_counts[action]
        regrets[t] = optimal_rewards[t] - reward
    return regrets

# 假设有4个节点
num_nodes = 4

# 生成时间序列数据
T = 1000
time_series_data = {i: np.random.normal(mean_execution_time, np.sqrt(var_execution_time), T) for i in range(num_nodes)}

# 提取奖励数据
rewards = {i: time_series_data[i] for i in range(num_nodes)}

# 运行epsilon-greedy算法
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]
results = {}
for epsilon in epsilons:
    regrets = epsilon_greedy(rewards, T, epsilon)
    results[epsilon] = regrets

# 绘制结果
def smooth_data(data, alpha=0.01):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]
    return smoothed

plt.figure(figsize=(10, 6))
for epsilon, regret in results.items():
    smoothed_regret = smooth_data(regret)
    plt.plot(smoothed_regret, label=f'Epsilon: {epsilon}')
plt.xlabel('Time Steps (T)')
plt.ylabel('Smoothed Instantaneous Regret')
plt.title('Epsilon-Greedy for different epsilon values')
plt.legend()
plt.grid(True)
plt.show()
