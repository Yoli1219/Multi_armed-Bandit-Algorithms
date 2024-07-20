import numpy as np
import matplotlib.pyplot as plt

# 定义节点参数
nodes = {
    'Node 1': {'mean': 52, 'variance': 5},
    'Node 2': {'mean': 48, 'variance': 5},
    'Node 3': {'mean': 47, 'variance': 5},
    'Node 4': {'mean': 49, 'variance': 5},
    'Node 5': {'mean': 50, 'variance': 5},
    'Node 6': {'mean': 55, 'variance': 5},
    'Node 7': {'mean': 60, 'variance': 5},
    'Node 8': {'mean': 53, 'variance': 5},
    'Node 9': {'mean': 54, 'variance': 5},
    'Node 10': {'mean': 58, 'variance': 5}
}

# Parameters
N = len(nodes)  # Number of arms
T = 1000  # Number of trials
epsilon = 0.1  # Exploration rate
alpha = 0.5  # Parameter for reward calculation

# Extract true means and variances from nodes
true_means = np.array([nodes[node]['mean'] for node in nodes])
true_variances = np.array([nodes[node]['variance'] for node in nodes])

# Function to generate rewards
def generate_rewards(mean, variance, T):
    return np.random.normal(loc=mean, scale=np.sqrt(variance), size=T)

# Function to choose an arm using epsilon-greedy strategy
def choose_arm(epsilon, Q_values):
    if np.random.random() < epsilon:
        # Exploration
        return np.random.randint(N)
    else:
        # Exploitation
        return np.argmax(Q_values)

# Generate rewards for each node
rewards = np.array([generate_rewards(nodes[node]['mean'], nodes[node]['variance'], T) for node in nodes])

# Initialize Q-values, counts, and lists to store results
Q_values = np.zeros(N)
counts = np.zeros(N)
top1_picks = np.zeros(T)
top2_picks = np.zeros(T)
top5_picks = np.zeros(T)
regret_list = []
actions = []

# Identify top nodes based on highest rewards
optimal_nodes = np.argsort(-true_means)
top1_node = optimal_nodes[0]
top2_nodes = optimal_nodes[:2]
top5_nodes = optimal_nodes[:5]

# Optimal reward for regret calculation
optimal_reward = np.exp(-1 / alpha)

# Epsilon-greedy algorithm with Q-value updates
for t in range(T):
    arm = choose_arm(epsilon, Q_values)
    reward = rewards[arm, t]

    # Update Q-value
    counts[arm] += 1
    Q_values[arm] += (reward - Q_values[arm]) / counts[arm]

    # Calculate regret and accuracy
    regret = optimal_reward - reward
    is_top1 = (arm == top1_node)
    is_top2 = (arm in top2_nodes)
    is_top5 = (arm in top5_nodes)

    # Store results
    regret_list.append(regret)
    actions.append(arm)
    top1_picks[t] = is_top1
    top2_picks[t] = is_top2
    top5_picks[t] = is_top5

# 计算最终的Top-1、Top-2和Top-5的准确性
top1_accuracy = np.sum(top1_picks) / T
top2_accuracy = np.sum(top2_picks) / T
top5_accuracy = np.sum(top5_picks) / T

# 输出Top-1、Top-2和Top-5的准确性
print(f"Top-1 Accuracy: {top1_accuracy:.2f}")
print(f"Top-2 Accuracy: {top2_accuracy:.2f}")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}")

# Plot the distribution of chosen nodes over time
plt.figure(figsize=(12, 8))
plt.plot(actions)
plt.xlabel('Time')
plt.ylabel('Chosen Node')
plt.yticks(ticks=range(N), labels=[f'Node {i + 1}' for i in range(N)])
plt.title('Epsilon-Greedy Node Selection Over Time')
plt.grid(True)
plt.show()

# Plot cumulative regret
plt.figure(figsize=(12, 8))
plt.plot(np.cumsum(regret_list))
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret Over Time')
plt.grid(True)
plt.show()

# Plot picks over time for Top-1, Top-2, and Top-5
plt.figure(figsize=(12, 8))
plt.plot(np.cumsum(top1_picks), label='Top-1 Picks')
plt.plot(np.cumsum(top2_picks), label='Top-2 Picks')
plt.plot(np.cumsum(top5_picks), label='Top-5 Picks')
plt.xlabel('Time')
plt.ylabel('Number of Picks')
plt.title('Top-K Picks Over Time')
plt.legend()
plt.grid(True)
plt.show()
