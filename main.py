import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Parameters
N_values = [10, 20, 50]  # Number of nodes
T = 1000  # Number of latency values per node
# 读取CSV文件
df = pd.read_csv('VM.csv')

# 显示前几行数据
print(df.head())

# 显示基本信息
print(df.info())

# 获取延迟数据
latencies = [df['Execution Time'].dropna().values]

# 检查延迟数据
print("延迟数据:", latencies[:5])  # 仅显示前5个数据

# Function to convert latency to rewards
def convert_latency_to_rewards(latencies, alpha):
    rewards = [np.exp(-alpha * lat) for lat in latencies]
    return rewards


# Epsilon-greedy algorithm
def epsilon_greedy(rewards, epsilon, T):
    N = len(rewards)
    q_values = np.zeros(N)
    n_arms = np.zeros(N)
    total_reward = 0
    rewards_history = []
    regret_history = []

    best_mean_reward = max([np.mean(r) for r in rewards])

    for t in range(T):
        if np.random.rand() < epsilon:
            arm = np.random.choice(N)
        else:
            arm = np.argmax(q_values)

        reward = rewards[arm][t]
        n_arms[arm] += 1
        q_values[arm] += (reward - q_values[arm]) / n_arms[arm]
        total_reward += reward

        rewards_history.append(total_reward)
        regret = (t + 1) * best_mean_reward - total_reward
        regret_history.append(regret)

    return rewards_history, regret_history


# Run epsilon-greedy for different values of epsilon
epsilons = [0.1, 0.2, 0.5]
alphas = [1, 2, 0.5]


def calculate_top_k_accuracy(rewards, q_values, k):
    sorted_indices = np.argsort(q_values)[::-1]
    best_k_indices = sorted_indices[:k]
    actual_best_indices = np.argsort([np.mean(r) for r in rewards])[::-1][:k]
    accuracy = len(set(best_k_indices).intersection(set(actual_best_indices))) / k
    return accuracy


# Calculate top-k accuracy
top_k_values = [1, 5, 10]


# Plotting function
def plot_regret(results, title):
    for alpha in alphas:
        for N in N_values:
            plt.figure(figsize=(14, 8))
            for epsilon in epsilons:
                _, regret = results[alpha][N][epsilon]
                plt.plot(regret, label=f'Epsilon: {epsilon}')
            plt.xlabel('Time')
            plt.ylabel('Regret')
            plt.title(f'{title} - Alpha: {alpha}, Nodes: {N}')
            plt.legend()
            plt.show()


# Generate latency data for nodes
def generate_latency_data(N, close=True):
    if close:
        rates = np.random.uniform(2.9, 3.1, N)  # Close latencies
    else:
        rates = np.random.uniform(3, 11, N)  # Far latencies
    latencies = [np.random.exponential(1 / rate, T) for rate in rates]
    return rates, latencies


# Generate data for different N values
data_close = {N: generate_latency_data(N, close=True) for N in N_values}
data_far = {N: generate_latency_data(N, close=False) for N in N_values}

# Convert for different alpha values
rewards_close = {alpha: {N: convert_latency_to_rewards(data_close[N][1], alpha) for N in N_values} for alpha in alphas}
rewards_far = {alpha: {N: convert_latency_to_rewards(data_far[N][1], alpha) for N in N_values} for alpha in alphas}

results_close = {
    alpha: {N: {epsilon: epsilon_greedy(rewards_close[alpha][N], epsilon, T) for epsilon in epsilons} for N in N_values}
    for alpha in alphas}
results_far = {
    alpha: {N: {epsilon: epsilon_greedy(rewards_far[alpha][N], epsilon, T) for epsilon in epsilons} for N in N_values}
    for alpha in alphas}

accuracy_close = {alpha: {N: {
    epsilon: {k: calculate_top_k_accuracy(rewards_close[alpha][N], results_close[alpha][N][epsilon][0], k) for k in
              top_k_values} for epsilon in epsilons} for N in N_values} for alpha in alphas}
accuracy_far = {alpha: {N: {
    epsilon: {k: calculate_top_k_accuracy(rewards_far[alpha][N], results_far[alpha][N][epsilon][0], k) for k in
              top_k_values} for epsilon in epsilons} for N in N_values} for alpha in alphas}

# Plot regrets for close and far scenarios
plot_regret(results_close, 'Close Latency Rates')
plot_regret(results_far, 'Far Latency Rates')

# Display top-k accuracy
df_accuracy_close = pd.DataFrame(accuracy_close)
df_accuracy_far = pd.DataFrame(accuracy_far)

print("Top-K Accuracy Close:")
print(df_accuracy_close)
print("\nTop-K Accuracy Far:")
print(df_accuracy_far)
