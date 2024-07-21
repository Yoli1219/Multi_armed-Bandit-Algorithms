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
thetas = [0.1, 0.5, 0.8]  # List of AR(1) coefficients

# Extract true means and variances from nodes
true_means = np.array([nodes[node]['mean'] for node in nodes])
true_variances = np.array([nodes[node]['variance'] for node in nodes])

# Function to generate AR(1) rewards
def generate_ar1_rewards(mean, variance, T, theta):
    rewards = np.zeros(T)
    rewards[0] = np.random.normal(loc=mean, scale=np.sqrt(variance))
    for t in range(1, T):
        rewards[t] = theta * rewards[t - 1] + (1 - theta) * mean + np.random.normal(scale=np.sqrt(variance))
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

for theta in thetas:
    # Generate AR(1) rewards for each node
    rewards = np.array([generate_ar1_rewards(nodes[node]['mean'], nodes[node]['variance'], T, theta) for node in nodes])

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

    # Calculate accuracy
    top1_accuracy = np.sum(np.array(actions) == top1_node) / T
    top2_accuracy = np.sum(np.isin(actions, top2_nodes)) / T
    top5_accuracy = np.sum(np.isin(actions, top5_nodes)) / T

    # Store results for this theta
    key = f"theta={theta}"
    results[key] = {
        'actions': actions,
        'regret_list': regret_list,
        'top1_accuracy': top1_accuracy,
        'top2_accuracy': top2_accuracy,
        'top5_accuracy': top5_accuracy,
    }

    # 输出Top-1、Top-2和Top-5的准确性
    print(f"Theta: {theta}")
    print(f"Top-1 Accuracy: {results[key]['top1_accuracy']:.2f}")
    print(f"Top-2 Accuracy: {results[key]['top2_accuracy']:.2f}")
    print(f"Top-5 Accuracy: {results[key]['top5_accuracy']:.2f}")

    # Plot the distribution of chosen nodes over time
    plt.figure(figsize=(12, 8))
    plt.plot(results[key]['actions'])
    plt.xlabel('Time')
    plt.ylabel('Chosen Node')
    plt.yticks(ticks=range(N), labels=[f'Node {i + 1}' for i in range(N)])
    plt.title(f'Adaptive Epsilon-Greedy Node Selection Over Time (theta={theta})')
    plt.grid(True)
    plt.show()

    # Plot cumulative regret
    plt.figure(figsize=(12, 8))
    plt.plot(np.cumsum(results[key]['regret_list']))
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    plt.title(f'Cumulative Regret Over Time (theta={theta})')
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
    plt.title(f'Top-K Picks Over Time (theta={theta})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot single trial regret
    plt.figure(figsize=(12, 8))
    plt.plot(results[key]['regret_list'])
    plt.xlabel('Time')
    plt.ylabel('Regret')
    plt.title(f'Single Trial Regret Over Time (theta={theta})')
    plt.grid(True)
    plt.show()

# Plot Top-K accuracy for different thetas
plt.figure(figsize=(12, 8))
top1_accuracies = [results[f"theta={theta}"]['top1_accuracy'] for theta in thetas]
top2_accuracies = [results[f"theta={theta}"]['top2_accuracy'] for theta in thetas]
top5_accuracies = [results[f"theta={theta}"]['top5_accuracy'] for theta in thetas]

bar_width = 0.2
r1 = np.arange(len(thetas))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.bar(r1, top1_accuracies, color='b', width=bar_width, edgecolor='grey', label='Top-1')
plt.bar(r2, top2_accuracies, color='r', width=bar_width, edgecolor='grey', label='Top-2')
plt.bar(r3, top5_accuracies, color='g', width=bar_width, edgecolor='grey', label='Top-5')

plt.xlabel('Theta')
plt.ylabel('Accuracy')
plt.title('Top-K Accuracy for Different Thetas')
plt.xticks([r + bar_width for r in range(len(thetas))], thetas)
plt.legend()
plt.grid(True)
plt.show()
