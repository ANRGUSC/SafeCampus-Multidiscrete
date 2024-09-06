import matplotlib.pyplot as plt
import seaborn as sns

# Data for DQN
dqn_data = {
    "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "optimal_y": [0, 25, 25, 25, 50, 100],
    "optimal_x": [4, 13, 27, 42, 72, 100]
}

# Data for Q-learning
q_learning_data = {
    "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "optimal_y": [0, 25, 25, 25, 50, 100],
    "optimal_x": [20, 20, 28, 47, 72, 100]
}

# Data for Myopic agent
myopic_data = {
    "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "optimal_y": [0, 25, 25, 25.0, 50, 100],
    "optimal_x": [20, 20, 20, 44.82, 69, 100]
}

# Colors for each agent
colors_dqn = sns.color_palette("Blues", len(dqn_data["alpha"]))
colors_q_learning = sns.color_palette("Greens", len(q_learning_data["alpha"]))
colors_myopic = sns.color_palette("Reds", len(myopic_data["alpha"]))

plt.figure(figsize=(8, 6))

# Plot DQN data with larger bubble sizes
for i, alpha in enumerate(dqn_data["alpha"]):
    plt.scatter(
        dqn_data["optimal_y"][i], dqn_data["optimal_x"][i],
        s=alpha * 800, color=colors_dqn[i], edgecolor='black', label=f'DQN Alpha {alpha}' if i == 0 else ""
    )
    # Add alpha value as text on the right side of the bubble
    plt.text(dqn_data["optimal_y"][i] + 2, dqn_data["optimal_x"][i], f'{alpha}', fontsize=10, ha='left')

# Plot Q-learning data with larger bubble sizes
for i, alpha in enumerate(q_learning_data["alpha"]):
    plt.scatter(
        q_learning_data["optimal_y"][i], q_learning_data["optimal_x"][i],
        s=alpha * 800, color=colors_q_learning[i], edgecolor='black', label=f'Q-learning Alpha {alpha}' if i == 0 else ""
    )
    # Add alpha value as text on the right side of the bubble
    plt.text(q_learning_data["optimal_y"][i] + 2, q_learning_data["optimal_x"][i], f'{alpha}', fontsize=10, ha='left')

# Plot Myopic data with larger bubble sizes
for i, alpha in enumerate(myopic_data["alpha"]):
    plt.scatter(
        myopic_data["optimal_y"][i], myopic_data["optimal_x"][i],
        s=alpha * 800, color=colors_myopic[i], edgecolor='black', label=f'Myopic Alpha {alpha}' if i == 0 else ""
    )
    # Add alpha value as text on the right side of the bubble
    plt.text(myopic_data["optimal_y"][i] + 2, myopic_data["optimal_x"][i], f'{alpha}', fontsize=10, ha='left')

# Adding a legend for agents
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='DQN', markerfacecolor='blue', markersize=10, markeredgecolor='black'),
    plt.Line2D([0], [0], marker='o', color='w', label='Q-learning', markerfacecolor='green', markersize=10, markeredgecolor='black'),
    plt.Line2D([0], [0], marker='o', color='w', label='Myopic', markerfacecolor='red', markersize=10, markeredgecolor='black')
]

plt.legend(handles=legend_elements, loc='upper left')

plt.title('Trade-off Between Optimal X and Y for Different Alpha Values (Bubble Size ~ Alpha)')
plt.xlabel('Optimal Y (Allowed)')
plt.ylabel('Optimal X (Infected)')
plt.grid(True)

# Save and show the plot
plt.tight_layout()
plt.savefig('tradeoff_optimal_xy_bubble_plot_larger_markers.png')
plt.show()
