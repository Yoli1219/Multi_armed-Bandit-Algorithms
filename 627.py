import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# 导入数据文件
file_path = os.path.expanduser("~/Desktop/Adaptive_Epsilon_Greedy/VM.csv")
data = pd.read_csv(file_path)


# 假设我们有10个节点
num_nodes = 10

# 随机生成不同的速率值（假设在0.1到1.0之间）
rate_values = np.random.uniform(0.1, 1.0, num_nodes)

# 基于速率值生成每个节点的T值
latency_data = data['Execution Time'].values
T_values = {}

for i in range(num_nodes):
    node_rate = rate_values[i]
    T_values[f'Node_{i + 1}'] = np.random.exponential(scale=1 / node_rate, size=len(latency_data))

# 将T值和执行时间合并到一个DataFrame
nodes_df = pd.DataFrame(T_values)
nodes_df['Execution Time'] = latency_data

# 不同的初始epsilon值
initial_epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# 窗口大小
window_size = 50

# 创建图像
plt.figure(figsize=(12, 6))

for initial_epsilon in initial_epsilons:
    # 初始化Adaptive Epsilon Greedy算法的参数
    decay_factor = 0.001
    num_steps = len(latency_data)
    Q = np.zeros(num_nodes)
    N = np.zeros(num_nodes)
    actual_rewards = np.zeros(num_steps)
    optimal_rewards = np.zeros(num_steps)

    # 自适应epsilon贪婪算法
    for step in range(num_steps):
        epsilon = initial_epsilon / (1 + decay_factor * step)

        if np.random.rand() < epsilon:
            action = np.random.choice(num_nodes)
        else:
            action = np.argmax(Q)

        # 获取当前动作的奖励
        reward = T_values[f'Node_{action + 1}'][step]
        actual_rewards[step] = reward

        # 更新期望奖励和选择次数
        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

        # 计算最优奖励
        optimal_action = np.argmax([np.mean(T_values[f'Node_{i + 1}']) for i in range(num_nodes)])
        optimal_rewards[step] = T_values[f'Node_{optimal_action + 1}'][step]

    # 计算单次遗憾
    single_step_regret = optimal_rewards - actual_rewards

    # 使用滑动平均法平滑曲线
    single_step_regret_smooth = np.convolve(single_step_regret, np.ones(window_size) / window_size, mode='valid')

    # 绘制平滑后的单次遗憾曲线
    plt.plot(single_step_regret_smooth, label=f'Initial Epsilon = {initial_epsilon}')

plt.xlabel('Steps')
plt.ylabel('Single Step Regret')
plt.title('Smoothed Single Step Regret Curves for Different Initial Epsilon Values')
plt.legend()
plt.grid(True)
plt.show()