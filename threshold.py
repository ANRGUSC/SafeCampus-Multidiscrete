import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Data for DQN
dqn_data = {
    "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "mean_allowed": [2, 29, 40, 63, 79, 100],
    "mean_infected": [0, 4, 8, 25, 44, 79]
}

# Data for Q-learning
q_learning_data = {
    "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "mean_allowed": [6, 28, 43, 61, 80, 99],
    "mean_infected": [0, 4, 10, 28, 45, 77]
}

# Data for Myopic agent
myopic_data = {
    "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "mean_allowed": [11, 30, 46, 64, 82, 97],
    "mean_infected": [1, 4, 11, 23, 46, 70]
}

# Colors for each alpha value using a color palette
alpha_values = dqn_data["alpha"]
colors = sns.color_palette("husl", len(alpha_values))

plt.figure(figsize=(8, 6))

# Plot DQN data with gray lines
sns.lineplot(
    x=dqn_data["mean_allowed"], y=dqn_data["mean_infected"],
    label='DQN', linestyle='-', linewidth=2, color='gray'
)

# Plot Q-learning data with gray lines
sns.lineplot(
    x=q_learning_data["mean_allowed"], y=q_learning_data["mean_infected"],
    label='Q-learning', linestyle='-.', linewidth=2, color='gray'
)

# Plot myopic data with gray lines
sns.lineplot(
    x=myopic_data["mean_allowed"], y=myopic_data["mean_infected"],
    label='Myopic', linestyle='--', linewidth=2, color='gray'
)

# Add colored markers for each alpha value to differentiate
for i, alpha in enumerate(alpha_values):
    # Scatter for DQN
    plt.scatter(
        dqn_data["mean_allowed"][i], dqn_data["mean_infected"][i],
        color=colors[i], s=100, edgecolor='black', label=f'Alpha {alpha}' if i == 0 else ""
    )
    # Scatter for Q-learning
    plt.scatter(
        q_learning_data["mean_allowed"][i], q_learning_data["mean_infected"][i],
        color=colors[i], s=100, edgecolor='black'
    )
    # Scatter for Myopic
    plt.scatter(
        myopic_data["mean_allowed"][i], myopic_data["mean_infected"][i],
        color=colors[i], s=100, edgecolor='black'
    )

# Adding a custom legend for the alpha values
legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Alpha {alpha}',
                          markerfacecolor=colors[i], markersize=10, markeredgecolor='black') for i, alpha in enumerate(alpha_values)]
legend_elements += [Line2D([0], [0], color='gray', linestyle='-', label='DQN'),
                    Line2D([0], [0], color='gray', linestyle='-.', label='Q-learning'),
                    Line2D([0], [0], color='gray', linestyle='--', label='Myopic')]

plt.legend(handles=legend_elements, title="Legend", loc="upper left")

plt.title('Trade-off Between Different Alpha Values')
plt.xlabel('Mean Allowed')
plt.ylabel('Mean Infected')
plt.grid(True)

# Save and show the plot
plt.tight_layout()
plt.savefig('tradeoff_alpha_values_colored_markers.png')
plt.show()
