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
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations
import matplotlib.patches as mpatches

# policy matrix


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import itertools


def visualize_all_states(model, all_states, run_name, num_courses, max_episodes, alpha, results_subdirectory,
                         students_per_course):
    method_name = "viz all states"
    file_paths = []
    colors = ['#a0b1ba', '#00b000', '#009ade']  # Light Red, Light Blue, Green
    color_map = {0: colors[0], 1: colors[1], 2: colors[2]}

    fig, axes = plt.subplots(1, num_courses, figsize=(5 * num_courses, 5), squeeze=False)
    fig.suptitle(f'{run_name}', fontsize=8)

    for course in range(num_courses):
        x_values = np.linspace(0, 1, 10)  # 10 values for community risk
        y_values = np.linspace(0, students_per_course[course], 10)  # 10 values for infected students
        xx, yy = np.meshgrid(x_values, y_values)

        x_flat = xx.flatten()
        y_flat = yy.flatten()

        color_values = []
        for i in range(len(x_flat)):
            state = [0] * course + [y_flat[i]] + [0] * (num_courses - course - 1) + [x_flat[i] * 100]
            adjusted_state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = model(adjusted_state)
                action = q_values[0, course::num_courses].argmax().item()
            color_values.append(color_map[action])

        ax = axes[0, course]
        scatter = ax.scatter(x_flat, y_flat, c=color_values, s=100, marker='s')

        ax.set_xlabel('Community Risk')
        ax.set_ylabel(f'Infected students in Course {course + 1}')
        ax.set_title(f'Course {course + 1}\nTotal Students: {students_per_course[course]}')
        ax.grid(False)

        # Add padding to y-axis
        y_padding = students_per_course[course] * 0.05  # 5% padding
        ax.set_ylim(-y_padding, students_per_course[course] + y_padding)

        # Add padding to x-axis
        ax.set_xlim(-0.05, 1.05)

        # Set y-ticks to show appropriate scale
        ax.set_yticks(np.linspace(0, students_per_course[course], 5))
        ax.set_yticklabels([f'{int(y)}' for y in ax.get_yticks()])

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
# def visualize_all_states(model, all_states, run_name, num_courses, max_episodes, alpha, results_subdirectory):
#     file_paths = []
#     method_name = "viz all states"
#
#     # Create color map
#     num_actions = 3  # Assuming 3 actions: 0%, 50%, 100%
#     colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, num_actions))
#     c = ListedColormap(colors)
#
#     input_dim = model.encoder[0].in_features
#     marker_size = 300  # Increased marker size
#
#     if input_dim == 2:
#         # If the model only takes 2 inputs, we assume it's [infected, community_risk]
#         x_values = [state[0] for state in all_states]
#         y_values = [state[-1] for state in all_states]
#
#         actions = {}
#         for state in all_states:
#             with torch.no_grad():
#                 q_values = model(torch.FloatTensor(state[:input_dim]).unsqueeze(0))
#                 action = q_values.max(1)[1].item()
#             actions[tuple(state)] = action
#
#         print("actions", actions)
#         colors = [actions[tuple(state)] for state in all_states]
#
#         plt.figure(figsize=(12, 10))
#         scatter = plt.scatter(x_values, y_values, c=colors, s=marker_size, marker='s', alpha=0.7, cmap=c)
#         plt.title(f"{method_name} - {run_name} (Infected vs Community Risk)")
#         plt.xlabel("Infected students")
#         plt.ylabel("Community risk")
#
#         legend_labels = ['Allow no one', '50% allowed', 'Allow everyone']
#         legend_handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10) for color in
#                           c.colors[:3]]
#         plt.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-large')
#
#         file_name = f"{max_episodes}-{method_name}-{run_name}-{alpha}-infected_vs_community_risk.png"
#         file_path = f"{results_subdirectory}/{file_name}"
#         plt.savefig(file_path, bbox_inches='tight')
#         plt.close()
#         file_paths.append(file_path)
#
#     else:
#         # Visualize each course vs community risk
#         for course in range(num_courses):
#             actions = {}
#             for state in all_states:
#                 adjusted_state = state[:input_dim]  # Truncate state to match model input
#                 with torch.no_grad():
#                     q_values = model(torch.FloatTensor(adjusted_state).unsqueeze(0))
#                     action = q_values.max(1)[1].item()
#                 actions[tuple(state)] = action
#
#             x_values = [state[course] for state in all_states]
#             y_values = [state[-1] for state in all_states]  # Community risk is last
#             colors = [actions[tuple(state)] for state in all_states]
#
#             plt.figure(figsize=(12, 10))
#             scatter = plt.scatter(x_values, y_values, c=colors, s=marker_size, marker='s', alpha=0.7, cmap=c)
#             plt.title(f"{method_name} - {run_name} (Course {course + 1} vs Community Risk)")
#             plt.xlabel(f"Infected students (Course {course + 1})")
#             plt.ylabel("Community risk")
#
#             legend_labels = ['Allow no one', '50% allowed', 'Allow everyone']
#             legend_handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10) for
#                               color in c.colors[:3]]
#             plt.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-large')
#
#             file_name = f"{max_episodes}-{method_name}-{run_name}-{alpha}-course_{course + 1}_vs_community_risk.png"
#             file_path = f"{results_subdirectory}/{file_name}"
#             plt.savefig(file_path, bbox_inches='tight')
#             plt.close()
#             file_paths.append(file_path)
#
#         # Visualize pairs of courses (optional, might be too many plots for many courses)
#         for course1, course2 in itertools.combinations(range(num_courses), 2):
#             actions = {}
#             for state in all_states:
#                 adjusted_state = state[:input_dim]  # Truncate state to match model input
#                 with torch.no_grad():
#                     q_values = model(torch.FloatTensor(adjusted_state).unsqueeze(0))
#                     action = q_values.max(1)[1].item()
#                 actions[tuple(state)] = action
#
#             x_values = [state[course1] for state in all_states]
#             y_values = [state[course2] for state in all_states]
#             colors = [actions[tuple(state)] for state in all_states]
#
#             plt.figure(figsize=(12, 10))
#             scatter = plt.scatter(x_values, y_values, c=colors, s=marker_size, marker='s', alpha=0.7, cmap=c)
#             plt.title(f"{method_name} - {run_name} (Course {course1 + 1} vs Course {course2 + 1})")
#             plt.xlabel(f"Infected students (Course {course1 + 1})")
#             plt.ylabel(f"Infected students (Course {course2 + 1})")
#
#             legend_labels = ['Allow no one', '50% allowed', 'Allow everyone']
#             legend_handles = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10) for
#                               color in c.colors[:3]]
#             plt.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-large')
#
#             file_name = f"{max_episodes}-{method_name}-{run_name}-{alpha}-course_{course1 + 1}_vs_course_{course2 + 1}.png"
#             file_path = f"{results_subdirectory}/{file_name}"
#             plt.savefig(file_path, bbox_inches='tight')
#             plt.close()
#             file_paths.append(file_path)
#
#     return file_paths
def visualize_q_table(q_table, results_subdirectory, episode):
    method_name = "viz q table"
    plt.figure(figsize=(10, 10))
    sns.heatmap(q_table, cmap="YlGnBu", annot=True, fmt=".2f")
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
    # print("num bins", num_bins, "rewars per episode", len(rewards_per_episode), "binned rewards var", len(binned_rewards_var))


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
    # Calculate explained variance for each episode\
    # Ensure predicted_rewards has compatible shape
    predicted_rewards = np.squeeze(predicted_rewards)
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
#     # Convert states to tuples if they're not already
#     states = [tuple(state) for state in states]
#
#     # Create a dictionary of state: visit_count
#     state_visits = dict(zip(states, visit_counts))
#
#     # Sort states by visit count in descending order
#     sorted_states = sorted(state_visits.items(), key=lambda x: x[1], reverse=True)
#
#     # Take top 20 most visited states
#     top_n = 100
#     top_states = sorted_states[:top_n]
#
#     # Separate states and counts for plotting
#     states, counts = zip(*top_states)
#
#     # Create state labels
#     state_labels = [f"({s[0]:.1f}, {s[1]:.1f})" for s in states]
#
#     # Create a bar chart
#     plt.figure(figsize=(12, 6))
#     bars = plt.bar(range(len(counts)), counts)
#
#     # Customize the plot
#     plt.title(f'Top {top_n} Most Visited States (α={alpha})')
#     plt.xlabel('States')
#     plt.ylabel('Visitation Count')
#     plt.xticks(range(len(counts)), state_labels, rotation=90)
#
#     # Add value labels on top of each bar
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                  f'{height}',
#                  ha='center', va='bottom')
#
#     plt.tight_layout()
#
#     # Save the plot
#     file_path = f"{results_subdirectory}/states_visited_α_{alpha}.png"
#     plt.savefig(file_path)
#     plt.close()
#
#     return file_path
import ast


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
    num_infected_dims = state_size - 1  # Last dimension is community risk

    file_paths = []

    # Create plots for each pair of infected dimensions
    for dim1, dim2 in combinations(range(num_infected_dims), 2):
        x_coords = sorted(set(state[dim1] for state in valid_states))
        y_coords = sorted(set(state[dim2] for state in valid_states))
        grid = np.zeros((len(y_coords), len(x_coords)))

        # Fill the grid with visit counts
        for state, count in zip(valid_states, visit_counts):
            x_index = x_coords.index(state[dim1])
            y_index = y_coords.index(state[dim2])
            grid[y_index, x_index] += count

        plt.figure(figsize=(12, 10))
        plt.imshow(grid, cmap='plasma', interpolation='nearest', origin='lower')
        cbar = plt.colorbar(label='Visitation Count')
        cbar.ax.tick_params(labelsize=10)

        plt.title(f'State Visitation Heatmap (α={alpha}, Dims: {dim1+1} vs {dim2+1})', fontsize=16)
        plt.xlabel(f'Infected Students (Dim {dim1+1})', fontsize=14)
        plt.ylabel(f'Infected Students (Dim {dim2+1})', fontsize=14)
        plt.xticks(range(len(x_coords)), [f'{int(x)}' for x in x_coords], fontsize=10, rotation=45)
        plt.yticks(range(len(y_coords)), [f'{int(y)}' for y in y_coords], fontsize=10)
        plt.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        file_path = f"{results_subdirectory}/states_visited_heatmap_α_{alpha}_dims_{dim1+1}vs{dim2+1}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        file_paths.append(file_path)

    # Create plot for each infected dimension vs community risk
    for dim in range(num_infected_dims):
        x_coords = sorted(set(state[dim] for state in valid_states))
        y_coords = sorted(set(state[-1] for state in valid_states))  # Community risk
        grid = np.zeros((len(y_coords), len(x_coords)))

        for state, count in zip(valid_states, visit_counts):
            x_index = x_coords.index(state[dim])
            y_index = y_coords.index(state[-1])
            grid[y_index, x_index] += count

        plt.figure(figsize=(12, 10))
        plt.imshow(grid, cmap='plasma', interpolation='nearest', origin='lower')
        cbar = plt.colorbar(label='Visitation Count')
        cbar.ax.tick_params(labelsize=10)

        plt.title(f'State Visitation Heatmap (α={alpha}, Dim {dim+1} vs Community Risk)', fontsize=16)
        plt.xlabel(f'Infected Students (Dim {dim+1})', fontsize=14)
        plt.ylabel('Community Risk', fontsize=14)
        plt.xticks(range(len(x_coords)), [f'{int(x)}' for x in x_coords], fontsize=10, rotation=45)
        plt.yticks(range(len(y_coords)), [f'{int(y)}' for y in y_coords], fontsize=10)
        plt.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        file_path = f"{results_subdirectory}/states_visited_heatmap_α_{alpha}_dim_{dim+1}_vs_community_risk.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        file_paths.append(file_path)










    # # Extract the first two dimensions for x and y coordinates
    # x_coords = [state[0] for state in valid_states]
    # y_coords = [state[1] for state in valid_states]
    #
    # # Print debugging information
    # # print(f"Number of valid states: {len(valid_states)}")
    # # print(f"Sample parsed states: {valid_states[:5]}")
    # # print(f"Sample x_coords: {x_coords[:5]}")
    # # print(f"Sample y_coords: {y_coords[:5]}")
    #
    # # Create a 2D grid for the heatmap
    # x_unique = sorted(set(x_coords))
    # y_unique = sorted(set(y_coords))
    # grid = np.zeros((len(y_unique), len(x_unique)))
    #
    # # Fill the grid with visit counts
    # for state, count in state_visits.items():
    #     i = y_unique.index(state[1])
    #     j = x_unique.index(state[0])
    #     grid[i, j] += count  # Sum counts for states sharing the same first two dimensions
    #
    # # Print grid information
    # # print(f"Grid shape: {grid.shape}")
    # # print(f"Grid min: {np.min(grid)}, max: {np.max(grid)}")
    #
    # # Create a heatmap
    # plt.figure(figsize=(12, 10))
    # plt.imshow(grid, cmap='plasma', interpolation='nearest', origin='lower')
    # cbar = plt.colorbar(label='Visitation Count')
    # cbar.ax.tick_params(labelsize=10)
    #
    # # Customize the plot
    # plt.title(f'State Visitation Heatmap (α={alpha})', fontsize=16)
    # plt.xlabel('Infected Students', fontsize=14)
    # plt.ylabel('Community Risk', fontsize=14)
    # plt.xticks(range(len(x_unique)), [f'{x:.1f}' for x in x_unique], fontsize=10)
    # plt.yticks(range(len(y_unique)), [f'{y:.1f}' for y in y_unique], fontsize=10)
    # plt.grid(True, color='white', linestyle='-', linewidth=0.1, alpha=0.5)
    #
    # plt.tight_layout()
    #
    # # Save the plot
    # file_path = f"{results_subdirectory}/states_visited_heatmap_α_{alpha}.png"
    # plt.savefig(file_path, dpi=300, bbox_inches='tight')
    # plt.close()

    return file_paths


