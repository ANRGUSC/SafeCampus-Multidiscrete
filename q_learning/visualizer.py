import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import seaborn as sns
import matplotlib.patches as mpatches
import colorsys
import ast

def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.95) for x in range(n)]  # Adjusted the brightness for better distinction
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    return ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in RGB_tuples]

def visualize_all_states(q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory,
                         students_per_course):
    method_name = "viz all states"
    state_size = len(states[0])
    num_courses = len(students_per_course)

    file_paths = []

    # Determine the number of unique actions (for distinct colors)
    unique_actions = q_table.shape[1] // num_courses
    colors = generate_distinct_colors(unique_actions)
    color_map = {i: colors[i] for i in range(unique_actions)}

    fig, axes = plt.subplots(1, num_courses, figsize=(5 * num_courses, 5), squeeze=False)
    fig.suptitle(f'Tabular-{alpha}', fontsize=16)

    for course in range(num_courses):
        actions = {}
        for state in states:
            state_idx = all_states.index(str(state))
            action = np.argmax(q_table[state_idx])

            # Extract course-specific action
            course_action = action % unique_actions

            # Key: (infected for this course, community risk)
            infected = state[course]
            community_risk = state[-1]
            actions[(infected, community_risk)] = course_action

        x_values = []  # Community risk
        y_values = []  # Infected
        color_values = []

        for (infected, community_risk), action in actions.items():
            x_values.append(community_risk / 9)  # Normalize to 0-1 range
            y_values.append(infected * (students_per_course[course] / 9))  # Scale to actual student numbers
            color_values.append(color_map[action])

        ax = axes[0, course]
        scatter = ax.scatter(x_values, y_values, c=color_values, s=100, marker='s')

        ax.set_xlabel('Community Risk')
        ax.set_ylabel(f'Infected students in Course {course + 1}')
        ax.grid(False)  # Remove grid

        max_val = students_per_course[course]
        y_margin = max_val * 0.05  # 5% margin
        ax.set_ylim(-y_margin, max_val + y_margin)
        ax.set_xlim(-0.05, 1.05)

    # Create a custom legend with arbitrary colors
    legend_elements = [mpatches.Patch(facecolor=colors[i], label=f' {i * (100 // (unique_actions - 1))}%') for i in range(unique_actions)]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 0.5), fontsize='large')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.9, wspace=0.3)

    file_name = f"{max_episodes}-{method_name}-{run_name}-{alpha}_multi_course.png"
    file_path = f"{results_subdirectory}/{file_name}"
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    file_paths.append(file_path)

    return file_paths

def visualize_q_table(q_table, results_subdirectory, episode):
    method_name = "viz q table"
    plt.figure(figsize=(10, 10))
    sns.heatmap(q_table, cmap="YlGnBu", annot=False, fmt=".2f")
    plt.title(f'Q-Table at Episode {episode} - {method_name}')
    plt.xlabel('Actions')
    plt.ylabel('States')
    file_path = f"{results_subdirectory}/qtable-{method_name}-{episode}.png"
    plt.savefig(file_path)
    plt.close()


def visualize_variance_in_rewards(rewards, results_subdirectory, episode):
    method_name = "viz insights"

    bin_size = 500  # number of episodes per bin, adjust as needed
    num_bins = len(rewards) // bin_size

    # Prepare data
    bins = []
    binned_rewards = []
    for i in range(num_bins):
        start = i * bin_size
        end = start + bin_size
        bin_rewards = rewards[start:end]
        bins.extend([f"{start}-{end}"] * bin_size)
        binned_rewards.extend(bin_rewards)

    data = pd.DataFrame({"Bin": bins, "Reward": binned_rewards})

    # Plot Variance (Box Plot)
    plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    sns.boxplot(x='Bin', y='Reward', data=data)
    plt.title(f'Variance in Rewards - {method_name}')
    plt.xlabel('Episode Bin')
    plt.ylabel('Reward')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Ensure everything fits without overlapping

    file_path_boxplot = f"{results_subdirectory}/variance_in_rewards-{method_name}-{episode}.png"
    plt.savefig(file_path_boxplot)
    plt.close()
    return file_path_boxplot

    # Log the boxplot image to wandb



def visualize_variance_in_rewards_heatmap(rewards_per_episode, results_subdirectory, bin_size):
    num_bins = len(rewards_per_episode) // bin_size
    binned_rewards_var = [np.var(rewards_per_episode[i * bin_size: (i + 1) * bin_size]) for i in
                          range(len(rewards_per_episode) // bin_size)]


    # Reshape to a square since we're assuming num_bins is a perfect square
    side_length = int(np.sqrt(num_bins))
    reshaped_var = np.array(binned_rewards_var).reshape(side_length, side_length)

    plt.figure(figsize=(10, 6))
    sns.heatmap(reshaped_var, cmap='YlGnBu', annot=True, fmt=".2f")
    plt.title('Variance in Rewards per Bin')
    plt.xlabel('Bin Index')
    plt.ylabel('Bin Index')
    file_path_heatmap = f"{results_subdirectory}/variance_in_rewards_heatmap.png"
    plt.savefig(file_path_heatmap)
    plt.close()
    return file_path_heatmap




def visualize_explained_variance(actual_rewards, predicted_rewards, results_subdirectory, max_episodes):
    # Calculate explained variance for each episode
    explained_variances = []
    for episode in range(1, max_episodes + 1):
        actual = actual_rewards[:episode]
        predicted = predicted_rewards[:episode]
        residuals = np.array(actual) - np.array(predicted)
        if np.var(actual) == 0:  # Prevent division by zero
            explained_variance = np.nan
        else:
            explained_variance = 1 - np.var(residuals) / np.var(actual)
        explained_variances.append(explained_variance)

    # Visualize explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_episodes + 1), explained_variances)
    plt.title('Explained Variance over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Explained Variance')
    file_path = f"{results_subdirectory}/explained_variance.png"
    plt.savefig(file_path)
    plt.close()
    return file_path

def visualize_infected_vs_community_risk(inf_comm_dict, alpha, results_subdirectory):
    community_risk = inf_comm_dict['community_risk']
    infected = inf_comm_dict['infected']
    allowed = inf_comm_dict['allowed']
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # set the y-axis limit
    plt.ylim(0, 120)

    # Scatter plots
    plt.scatter(community_risk, infected, color='blue', label="Infected", alpha=alpha, s=60)
    plt.scatter(community_risk, allowed, color='red', label="Allowed", alpha=alpha, s=60)

    # Set the title and labels
    plt.title('Infected vs Community Risk with alpha = ' + str(alpha))
    plt.xlabel('Community Risk')
    plt.ylabel('Number of Students')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Adjust layout to accommodate the legend
    plt.tight_layout()
    file_path = f"{results_subdirectory}/infected_vs_community_risk.png"
    plt.savefig(file_path)
    plt.close()
    return file_path


def visualize_infected_vs_community_risk_table(inf_comm_dict, alpha, results_subdirectory):
    community_risk = inf_comm_dict['community_risk']
    infected = inf_comm_dict['infected']
    allowed = inf_comm_dict['allowed']

    # Combine the data into a list of lists
    data = list(zip(community_risk, infected, allowed))

    # Define the headers for the table
    headers = ["Community Risk", "Infected", "Allowed"]

    # Use the tabulate function to create a table
    table = tabulate(data, headers, tablefmt="pretty")

    # Define the title with alpha
    title = f'Infected vs Community Risk with alpha = {alpha}'

    # Create a Matplotlib figure and axis to render the table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')  # Turn off axis labels

    # Render the table
    ax.table(cellText=data, colLabels=headers, loc='center')

    # Add the title to the table
    ax.set_title(title, fontsize=14)

    # Save the table as an image
    file_path = f"{results_subdirectory}/infected_vs_community_risk_table.png"
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    return file_path

def states_visited_viz(states, visit_counts, alpha, results_subdirectory):
    def parse_state(state):
        if isinstance(state, (list, tuple)):
            try:
                return [float(x) for x in state]
            except ValueError:
                pass
        elif isinstance(state, str):
            try:
                evaluated_state = ast.literal_eval(state)
                if isinstance(evaluated_state, (list, tuple)):
                    return [float(x) for x in evaluated_state]
            except (ValueError, SyntaxError):
                print(f"Error parsing state: {state}")
        print(f"Unexpected state format: {state}")
        return None

    parsed_states = [parse_state(state) for state in states]
    valid_states = [state for state in parsed_states if state is not None]

    if not valid_states:
        print("Error: No valid states found after parsing")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Error: No valid states found after parsing", ha='center', va='center')
        plt.axis('off')
        error_path = f"{results_subdirectory}/states_visited_error_α_{alpha}.png"
        plt.savefig(error_path)
        plt.close()
        return [error_path]

    state_size = len(valid_states[0])
    num_infected_dims = state_size - 1

    file_paths = []

    for dim in range(num_infected_dims):
        x_coords = sorted(set(state[dim] for state in valid_states))
        y_coords = sorted(set(state[-1] for state in valid_states))
        grid = np.zeros((len(y_coords), len(x_coords)))

        for state, count in zip(valid_states, visit_counts):
            i = y_coords.index(state[-1])
            j = x_coords.index(state[dim])
            grid[i, j] += count

        plt.figure(figsize=(12, 10))
        plt.imshow(grid, cmap='plasma', interpolation='nearest', origin='lower')
        cbar = plt.colorbar(label='Visitation Count')
        cbar.ax.tick_params(labelsize=10)

        plt.title(f'State Visitation Heatmap (α={alpha}, Infected Dim: {dim + 1})', fontsize=16)
        plt.xlabel(f'Infected Students (Dim {dim + 1})', fontsize=14)
        plt.ylabel('Community Risk', fontsize=14)
        plt.xticks(range(len(x_coords)), [f'{int(x)}' for x in x_coords], fontsize=10, rotation=45)
        plt.yticks(range(len(y_coords)), [f'{int(y)}' for y in y_coords], fontsize=10)
        plt.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        file_path = f"{results_subdirectory}/states_visited_heatmap_α_{alpha}_infected_dim_{dim + 1}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        file_paths.append(file_path)

    return file_paths


