import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Data for DQN
dqn_data = {
    "alpha": [0.2, 0.4, 0.5],
    "optimal_y": [25.0, 25.0, 50.0],
    "optimal_x": [13.00, 42.00, 72.00]
}

# Data for Q-learning
q_learning_data = {
    "alpha": [0.2, 0.4, 0.5],
    "optimal_y": [25.0, 25.0, 50.0],
    "optimal_x": [20.00, 47.00, 72.00]
}

# Data for Myopic agent
myopic_data = {
    "alpha": [0.2, 0.4, 0.5],
    "optimal_y": [25.0, 25.0, 50.0],
    "optimal_x": [20.00, 44.82, 68.55]
}

# Colors for each alpha value using a color palette
alpha_values = dqn_data["alpha"]
colors = sns.color_palette("husl", len(alpha_values))

plt.figure(figsize=(8, 6))

# Plot DQN data with gray lines
sns.lineplot(
    x=dqn_data["optimal_y"], y=dqn_data["optimal_x"],
    label='DQN', linestyle='-', linewidth=2, color='gray'
)

# Plot Q-learning data with gray lines
sns.lineplot(
    x=q_learning_data["optimal_y"], y=q_learning_data["optimal_x"],
    label='Q-learning', linestyle='-.', linewidth=2, color='gray'
)

# Plot myopic data with gray lines
sns.lineplot(
    x=myopic_data["optimal_y"], y=myopic_data["optimal_x"],
    label='Myopic', linestyle='--', linewidth=2, color='gray'
)

# Add colored markers for each alpha value to differentiate
for i, alpha in enumerate(alpha_values):
    # Scatter for DQN
    plt.scatter(
        dqn_data["optimal_y"][i], dqn_data["optimal_x"][i],
        color=colors[i], s=100, edgecolor='black', label=f'Alpha {alpha}' if i == 0 else ""
    )
    # Scatter for Q-learning
    plt.scatter(
        q_learning_data["optimal_y"][i], q_learning_data["optimal_x"][i],
        color=colors[i], s=100, edgecolor='black'
    )
    # Scatter for Myopic
    plt.scatter(
        myopic_data["optimal_y"][i], myopic_data["optimal_x"][i],
        color=colors[i], s=100, edgecolor='black'
    )

# Adding a custom legend for the alpha values
legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Alpha {alpha}',
                          markerfacecolor=colors[i], markersize=10, markeredgecolor='black') for i, alpha in enumerate(alpha_values)]
legend_elements += [Line2D([0], [0], color='gray', linestyle='-', label='DQN'),
                    Line2D([0], [0], color='gray', linestyle='-.', label='Q-learning'),
                    Line2D([0], [0], color='gray', linestyle='--', label='Myopic')]

plt.legend(handles=legend_elements, title="Legend", loc="upper left")

plt.title('Trade-off Between Optimal X and Y for Different Alpha Values')
plt.xlabel('Optimal Y (Allowed)')
plt.ylabel('Optimal X (Infected)')
plt.grid(True)

# Save and show the plot
plt.tight_layout()
plt.savefig('tradeoff_optimal_xy_colored_markers.png')
plt.show()
