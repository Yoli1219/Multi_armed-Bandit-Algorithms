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
initial_epsilon = 0.1  # Initial exploration rate
decay_rate = 0.99  # Decay rate for epsilon

# Extract true means and variances from nodes
true_means = np.array([nodes[node]['mean'] for node in nodes])
true_variances = np.array([nodes[node]['variance'] for node in nodes])

# Function to generate IID rewards
def generate_iid_rewards(mean, variance, T):
    rewards = np.random.normal(loc=mean, scale=np.sqrt(variance), size=T)
    return rewards

# Function to choose an arm using adaptive epsilon-greedy strategy
def choose_arm(epsilon, Q_values):
    if np.random.random() < epsilon:
        # Exploration
        return np.random.randint(N)
    else:
        # Exploitation
        return np.argmax(Q_values)

# Initialize results storage
results = {}

# Generate IID rewards for each node
rewards = np.array([generate_iid_rewards(nodes[node]['mean'], nodes[node]['variance'], T) for node in nodes])

# Initialize Q-values, counts, and lists to store results
Q_values = np.zeros(N)
counts = np.zeros(N)
regret_list = []
actions = []

# Identify top nodes based on highest rewards
optimal_nodes = np.argsort(-true_means)
top1_node = optimal_nodes[0]
top2_nodes = optimal_nodes[:2]
top5_nodes = optimal_nodes[:5]

# Optimal reward for regret calculation
optimal_reward = true_means[top1_node]

# Adaptive epsilon-greedy algorithm with Q-value updates
epsilon = initial_epsilon
for t in range(T):
    arm = choose_arm(epsilon, Q_values)
    reward = rewards[arm, t]

    # Update Q-value
    counts[arm] += 1
    Q_values[arm] += (reward - Q_values[arm]) / counts[arm]

    # Decay epsilon
    epsilon *= decay_rate

    # Calculate regret and store actions
    regret = optimal_reward - reward
    regret_list.append(regret)
    actions.append(arm)

    # Debugging output for the first few steps
    if t < 10:
        print(f"Step {t}, Arm {arm}, Reward {reward:.2f}, Q-values {Q_values}, Epsilon {epsilon:.4f}")

# Calculate accuracy
top1_accuracy = np.sum(np.array(actions) == top1_node) / T
top2_accuracy = np.sum(np.isin(actions, top2_nodes)) / T
top5_accuracy = np.sum(np.isin(actions, top5_nodes)) / T

# Store results
results['adaptive'] = {
    'actions': actions,
    'regret_list': regret_list,
    'top1_accuracy': top1_accuracy,
    'top2_accuracy': top2_accuracy,
    'top5_accuracy': top5_accuracy,
}

# 输出Top-1、Top-2和Top-5的准确性
print(f"Adaptive Epsilon-Greedy Algorithm")
print(f"Top-1 Accuracy: {results['adaptive']['top1_accuracy']:.2f}")
print(f"Top-2 Accuracy: {results['adaptive']['top2_accuracy']:.2f}")
print(f"Top-5 Accuracy: {results['adaptive']['top5_accuracy']:.2f}")

# Plot the distribution of chosen nodes over time
plt.figure(figsize=(12, 8))
plt.plot(results['adaptive']['actions'])
plt.xlabel('Time')
plt.ylabel('Chosen Node')
plt.yticks(ticks=range(N), labels=[f'Node {i + 1}' for i in range(N)])
plt.title('Adaptive Epsilon-Greedy Node Selection Over Time')
plt.grid(True)
plt.show()

# Plot cumulative regret
plt.figure(figsize=(12, 8))
plt.plot(np.cumsum(results['adaptive']['regret_list']))
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret Over Time (Adaptive Epsilon-Greedy)')
plt.grid(True)
plt.show()

# Plot picks over time for Top-1, Top-2, and Top-5
plt.figure(figsize=(12, 8))
top1_picks = np.cumsum(np.array(actions) == top1_node)
top2_picks = np.cumsum(np.isin(actions, top2_nodes))
top5_picks = np.cumsum(np.isin(actions, top5_nodes))
plt.plot(top1_picks/T, label='Top-1 Picks')
plt.plot(top2_picks/T, label='Top-2 Picks')
plt.plot(top5_picks/T, label='Top-5 Picks')
plt.xlabel('Time')
plt.ylabel('Proportion of Picks')
plt.title('Top-K Picks Over Time (Adaptive Epsilon-Greedy)')
plt.legend()
plt.grid(True)
plt.show()

# Plot single trial regret
plt.figure(figsize=(12, 8))
plt.plot(results['adaptive']['regret_list'])
plt.xlabel('Time')
plt.ylabel('Regret')
plt.title('Single Trial Regret Over Time (Adaptive Epsilon-Greedy)')
plt.grid(True)
plt.show()
