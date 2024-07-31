import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
import wandb
import scipy.stats as stats
import pandas as pd
from tabulate import tabulate
import seaborn as sns
from collections import defaultdict
from itertools import combinations
import matplotlib.patches as mpatches


def visualize_all_states(q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory,
                         students_per_course):
    method_name = "viz all states"
    # print("students_per_course:", students_per_course)
    # print("states:", states)

    # Determine the number of dimensions
    state_size = len(states[0])
    num_courses = len(students_per_course)

    file_paths = []
    colors = ['#a0b1ba', '#00b000', '#009ade']  # Light Red, Light Blue, Light Green
    color_map = {0: colors[0], 1: colors[1], 2: colors[2]}

    fig, axes = plt.subplots(1, num_courses, figsize=(5 * num_courses, 5), squeeze=False)
    fig.suptitle(f'{run_name})', fontsize=16)

    for course in range(num_courses):
        actions = {}
        for state in states:
            state_idx = all_states.index(str(state))
            action = np.argmax(q_table[state_idx])

            # Extract course-specific action
            course_action = action % 3

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
        # ax.set_title(f'Course {course + 1}\nTotal Students: {students_per_course[course]}')
        ax.grid(False)  # Remove grid

        max_val = students_per_course[course]
        y_margin = max_val * 0.05  # 5% margin
        ax.set_ylim(-y_margin, max_val + y_margin)
        ax.set_xlim(-0.05, 1.05)

    # Create a custom legend
    legend_elements = [mpatches.Patch(facecolor=colors[i], label=f'Allow {i * 50}%') for i in range(3)]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=3, fontsize='large')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.9, wspace=0.3)

    file_name = f"{max_episodes}-{method_name}-{run_name}-{alpha}_multi_course.png"
    file_path = f"{results_subdirectory}/{file_name}"
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    plt.close()
    file_paths.append(file_path)

    return file_paths




#
# def visualize_all_states(q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory):
#     method_name = "viz all states"
#     actions = {}
#     for i in states:
#         action = np.argmax(q_table[all_states.index(str(i))])
#         actions[(i[0], i[1])] = action
#
#     x_values = []
#     y_values = []
#     colors = []
#     for k, v in actions.items():
#         x_values.append(k[0])
#         y_values.append(k[1])
#         colors.append(v)
#
#     c = ListedColormap(['red', 'green', 'blue'])
#
#     plt.figure(figsize=(10, 10))
#     scatter = plt.scatter(y_values, x_values, c=colors, s=500, marker='s', cmap=c)
#     plt.title(f"{method_name} - {run_name}")
#     plt.xlabel("Community risk")
#     plt.ylabel("Infected students")
#
#     # Create a legend with explicit labels
#     legend_labels = ['Allow no one', '50% allowed', 'Allow everyone']
#     legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in c.colors]
#     plt.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-large')
#
#     file_name = f"{max_episodes}-{method_name}-{run_name}-{alpha}.png"
#     file_path = f"{results_subdirectory}/{file_name}"
#     plt.savefig(file_path, bbox_inches='tight')  # Use bbox_inches='tight' to include the legend in the saved image
#     plt.close()
#     return file_path

# def visualize_all_states(q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory, students_per_course):
#     method_name = "viz all states"
#
#     # Determine the number of dimensions
#     state_size = len(states[0])
#     num_infected_dims = state_size - 1  # Last dimension is community risk
#
#     file_paths = []
#
#     colors = ['#FF9999', '#66B2FF', '#99FF99']  # Light Red, Light Blue, Light Green
#     color_map = {0: colors[0], 1: colors[1], 2: colors[2]}
#
#     for infected_dim in range(num_infected_dims):
#         actions = {}
#         for state in states:
#             action = np.argmax(q_table[all_states.index(str(state))])
#             actions[(state[infected_dim], state[-1])] = action  # (infected, community risk)
#
#         x_values = []  # Community risk
#         y_values = []  # Infected
#         color_values = []
#         for k, v in actions.items():
#             y_values.append(k[0])  # Infected dimension
#             x_values.append(k[1])  # Community risk
#             color_values.append(color_map[v])
#
#         total_students = students_per_course[infected_dim]
#         fig, ax = plt.subplots(figsize=(12, 10))
#         scatter = ax.scatter(x_values, y_values, c=color_values, s=500, marker='s')
#
#         # Adjust y-axis limits to show full markers
#         y_margin = total_students * 0.05  # 5% margin
#         ax.set_ylim(-y_margin, total_students + y_margin)
#
#         # Adjust x-axis limits to show full markers
#         ax.set_xlim(-0.05, 1.05)
#
#         # Create a custom legend
#         legend_elements = [mpatches.Patch(facecolor=colors[i], edgecolor='black', label=f'Allow {v}')
#                            for i, v in enumerate([0, 50, 100])]
#         fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05),
#                    ncol=3, fontsize='large')
#
#         plt.tight_layout()
#         plt.subplots_adjust(bottom=0.15, wspace=0.3)
#
#         file_name = f"{max_episodes}-{method_name}-{run_name}-{alpha}-infected_dim_{infected_dim + 1}.png"
#         file_path = f"{results_subdirectory}/{file_name}"
#         plt.savefig(file_path, bbox_inches='tight', dpi=300)
#         plt.close()
#
#         file_paths.append(file_path)
#
#     return file_paths

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
    print("num bins", num_bins, "rewars per episode", len(rewards_per_episode), "binned rewards var", len(binned_rewards_var))


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

# def states_visited_viz(states, visit_counts, alpha, results_subdirectory):
#     title = f'State Visitation Frequency During Training with alpha: = {alpha}'
#     # Create a bar chart
#     plt.figure(figsize=(10, 6))  # Adjust figure size as needed
#     plt.bar(states, visit_counts)
#
#     # Rotate x-axis labels if there are many states for better readability
#     plt.xticks(rotation=90)
#
#     # Add labels and title
#     plt.xlabel('State')
#     plt.ylabel('Visitation Count')
#     plt.title(title)
#
#     plt.tight_layout()
#     file_path = f"{results_subdirectory}/states_visited.png"
#     plt.savefig(file_path)
#     plt.close()
#     return file_path
# def states_visited_viz(states, visit_counts, alpha, results_subdirectory):
#     # Sort states and corresponding visit counts
#     sorted_indices = sorted(range(len(states)), key=lambda i: states[i])
#     sorted_states = [states[i] for i in sorted_indices]
#     sorted_visit_counts = [visit_counts[i] for i in sorted_indices]
#
#     title = f'State Visitation Frequency During Training with alpha: = {alpha}'
#
#     # Create a bar chart
#     plt.figure(figsize=(10, 6))  # Adjust figure size as needed
#     plt.bar(sorted_states, sorted_visit_counts)
#
#     # Rotate x-axis labels if there are many states for better readability
#     plt.xticks(rotation=90)
#
#     # Add labels and title
#     plt.xlabel('State')
#     plt.ylabel('Visitation Count')
#     plt.title(title)
#
#     plt.tight_layout()
#     file_path = f"{results_subdirectory}/states_visited.png"
#     plt.savefig(file_path)
#     plt.close()
#
#     return file_path
import ast
# def states_visited_viz(states, visit_counts, alpha, results_subdirectory):
#     print('Original states:', states)
#
#     def parse_state(state):
#         # print(f"Parsing state: {state} (type: {type(state)})")
#         if isinstance(state, (list, tuple)) and len(state) == 2:
#             try:
#                 return [float(x) for x in state]
#             except ValueError:
#                 pass
#         elif isinstance(state, str):
#             try:
#                 # Attempt to evaluate the string to convert it to a tuple or list
#                 evaluated_state = ast.literal_eval(state)
#                 if isinstance(evaluated_state, (list, tuple)) and len(evaluated_state) == 2:
#                     return [float(x) for x in evaluated_state]
#             except (ValueError, SyntaxError):
#                 print(f"Error parsing state: {state}")
#                 return None
#         else:
#             print(f"Unexpected state format: {state}")
#             return None
#
#     # Parse states
#     parsed_states = [parse_state(state) for state in states]
#     valid_states = [state for state in parsed_states if state is not None]
#
#     if not valid_states:
#         print("Error: No valid states found after parsing")
#         plt.figure(figsize=(10, 6))
#         plt.text(0.5, 0.5, "Error: No valid states found after parsing", ha='center', va='center')
#         plt.axis('off')
#         error_path = f"{results_subdirectory}/states_visited_error_α_{alpha}.png"
#         plt.savefig(error_path)
#         plt.close()
#         return error_path
#
#     # Create a dictionary of state: visit_count for valid states
#     state_visits = {tuple(state): count for state, count in zip(valid_states, visit_counts) if state is not None}
#
#     # Extract the first two dimensions for x and y coordinates
#     x_coords = [state[0] for state in valid_states]
#     y_coords = [state[1] for state in valid_states]
#
#     # Print debugging information
#     # print(f"Number of valid states: {len(valid_states)}")
#     # print(f"Sample parsed states: {valid_states[:5]}")
#     # print(f"Sample x_coords: {x_coords[:5]}")
#     # print(f"Sample y_coords: {y_coords[:5]}")
#
#     # Create a 2D grid for the heatmap
#     x_unique = sorted(set(x_coords))
#     y_unique = sorted(set(y_coords))
#     grid = np.zeros((len(y_unique), len(x_unique)))
#
#     # Fill the grid with visit counts
#     for state, count in state_visits.items():
#         i = y_unique.index(state[1])
#         j = x_unique.index(state[0])
#         grid[i, j] += count  # Sum counts for states sharing the same first two dimensions
#
#     # Print grid information
#     # print(f"Grid shape: {grid.shape}")
#     # print(f"Grid min: {np.min(grid)}, max: {np.max(grid)}")
#
#     # Create a heatmap
#     plt.figure(figsize=(12, 10))
#     plt.imshow(grid, cmap='plasma', interpolation='nearest', origin='lower')
#     cbar = plt.colorbar(label='Visitation Count')
#     cbar.ax.tick_params(labelsize=10)
#
#     # Customize the plot
#     plt.title(f'State Visitation Heatmap (α={alpha})', fontsize=16)
#     plt.xlabel('Infected Students', fontsize=14)
#     plt.ylabel('Community Risk', fontsize=14)
#     plt.xticks(range(len(x_unique)), [f'{int(x)}' for x in x_unique], fontsize=10, rotation=45)
#     plt.yticks(range(len(y_unique)), [f'{int(y)}' for y in y_unique], fontsize=10)
#     plt.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
#
#     plt.tight_layout()
#
#     # Save the plot
#     file_path = f"{results_subdirectory}/states_visited_heatmap_α_{alpha}.png"
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
#     plt.close()
#
#     return file_path

def states_visited_viz(states, visit_counts, alpha, results_subdirectory):
    # print('Original states:', states)

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
        # Create a 2D grid for the heatmap
        x_coords = sorted(set(state[dim] for state in valid_states))
        y_coords = sorted(set(state[-1] for state in valid_states))  # Community risk
        grid = np.zeros((len(y_coords), len(x_coords)))

        # Fill the grid with visit counts
        for state, count in zip(valid_states, visit_counts):
            i = y_coords.index(state[-1])  # Community risk
            j = x_coords.index(state[dim])  # Infected dimension
            grid[i, j] += count

        # Create a heatmap
        plt.figure(figsize=(12, 10))
        plt.imshow(grid, cmap='plasma', interpolation='nearest', origin='lower')
        cbar = plt.colorbar(label='Visitation Count')
        cbar.ax.tick_params(labelsize=10)

        # Customize the plot
        plt.title(f'State Visitation Heatmap (α={alpha}, Infected Dim: {dim + 1})', fontsize=16)
        plt.xlabel(f'Infected Students (Dim {dim + 1})', fontsize=14)
        plt.ylabel('Community Risk', fontsize=14)
        plt.xticks(range(len(x_coords)), [f'{int(x)}' for x in x_coords], fontsize=10, rotation=45)
        plt.yticks(range(len(y_coords)), [f'{int(y)}' for y in y_coords], fontsize=10)
        plt.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        # Save the plot
        file_path = f"{results_subdirectory}/states_visited_heatmap_α_{alpha}_infected_dim_{dim + 1}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        file_paths.append(file_path)

    return file_paths

