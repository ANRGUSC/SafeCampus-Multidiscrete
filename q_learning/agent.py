import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import itertools
from .utilities import load_config
from .visualizer import visualize_all_states, visualize_q_table, visualize_variance_in_rewards_heatmap, \
    visualize_explained_variance, visualize_variance_in_rewards, visualize_infected_vs_community_risk_table, states_visited_viz
import os
import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import random
import pandas as pd
import csv
from scipy import stats
import time
import torch
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.signal import argrelextrema
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
epsilon = 1e-10


# Define the neural network model for the Lyapunov function
class LyapunovNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super(LyapunovNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = F.softplus(self.fc3(out))  # Ensure positive output
        return out
# Instantiate the model
input_dim = 3  # Number of features
hidden_dim = 64  # Number of hidden units
output_dim = 1  # Output is a single value representing V(s)
model = LyapunovNet(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def V(s, theta):
    # Ensure that each element of s is a scalar or flatten it if it's a list
    s = [item[0] if isinstance(item, list) else item for item in s]
    assert len(s) == 3, f"Feature vector length mismatch: expected 3, got {len(s)}"
    return 0.5 * (theta[0] * s[0] ** 2 + theta[1] * s[1] ** 2 + theta[2] * s[2] ** 2)

def log_metrics_to_csv(file_path, metrics):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

class ExplorationRateDecay:
    def __init__(self, max_episodes, min_exploration_rate, initial_exploration_rate):
        self.max_episodes = max_episodes
        self.min_exploration_rate = min_exploration_rate
        self.initial_exploration_rate = initial_exploration_rate
        self.current_decay_function = 1  # Variable to switch between different decay functions

    def set_decay_function(self, decay_function_number):
        self.current_decay_function = decay_function_number

    def get_exploration_rate(self, episode):
        if self.current_decay_function == 1:  # Exponential Decay
            decay_factor = np.exp(-1 / self.max_episodes)
            exploration_rate = self.initial_exploration_rate
            exploration_rate *= decay_factor
            exploration_rate = max(exploration_rate, self.min_exploration_rate)

        elif self.current_decay_function == 2:  # Linear Decay
            exploration_rate = self.initial_exploration_rate - (
                        self.initial_exploration_rate - self.min_exploration_rate) * (episode / self.max_episodes)

        elif self.current_decay_function == 3:  # Polynomial Decay
            exploration_rate = self.initial_exploration_rate * (1 - episode / self.max_episodes) ** 2

        elif self.current_decay_function == 4:  # Inverse Time Decay
            exploration_rate = self.initial_exploration_rate / (1 + episode)

        elif self.current_decay_function == 5:  # Sine Wave Decay
            exploration_rate = self.min_exploration_rate + 0.5 * (
                        self.initial_exploration_rate - self.min_exploration_rate) * (
                                           1 + np.sin(np.pi * episode / self.max_episodes))

        elif self.current_decay_function == 6:  # Logarithmic Decay
            exploration_rate = self.initial_exploration_rate - (
                        self.initial_exploration_rate - self.min_exploration_rate) * np.log(episode + 1) / np.log(
                self.max_episodes + 1)

        elif self.current_decay_function == 7:  # Hyperbolic Tangent Decay
            exploration_rate = self.min_exploration_rate + 0.5 * (
                        self.initial_exploration_rate - self.min_exploration_rate) * (
                                           1 - np.tanh(episode / self.max_episodes))
        elif self.current_decay_function == 8:  # Square Root Decay
            exploration_rate = self.initial_exploration_rate * (1 - np.sqrt(episode / self.max_episodes))
        elif self.current_decay_function == 9:  # Stepwise Decay
            steps = 10
            step_size = (self.initial_exploration_rate - self.min_exploration_rate) / steps
            exploration_rate = self.initial_exploration_rate - (episode // (self.max_episodes // steps)) * step_size
        elif self.current_decay_function == 10:  # Inverse Square Root Decay
            exploration_rate = self.initial_exploration_rate / np.sqrt(episode + 1)
        elif self.current_decay_function == 11:  # Sigmoid Decay
            midpoint = self.max_episodes / 2
            smoothness = self.max_episodes / 10  # Adjust this divisor to change smoothness
            exploration_rate = self.min_exploration_rate + (
                        self.initial_exploration_rate - self.min_exploration_rate) / (
                                           1 + np.exp((episode - midpoint) / smoothness))
        elif self.current_decay_function == 12:  # Quadratic Decay
            exploration_rate = self.initial_exploration_rate * (1 - (episode / self.max_episodes) ** 2)
        elif self.current_decay_function == 13:  # Cubic Decay
            exploration_rate = self.initial_exploration_rate * (1 - (episode / self.max_episodes) ** 3)
        elif self.current_decay_function == 14:  # Sine Squared Decay
            exploration_rate = self.min_exploration_rate + (
                        self.initial_exploration_rate - self.min_exploration_rate) * np.sin(
                np.pi * episode / self.max_episodes)
        elif self.current_decay_function == 15:  # Cosine Squared Decay
            exploration_rate = self.min_exploration_rate + (
                        self.initial_exploration_rate - self.min_exploration_rate) * np.cos(
                np.pi * episode / self.max_episodes) ** 2
        elif self.current_decay_function == 16:  # Double Exponential Decay
            exploration_rate = self.initial_exploration_rate * np.exp(-np.exp(episode / self.max_episodes))
        elif self.current_decay_function == 17:  # Log-Logistic Decay
            exploration_rate = self.min_exploration_rate + (
                        self.initial_exploration_rate - self.min_exploration_rate) / (1 + np.log(episode + 1))
        elif self.current_decay_function == 18:  # Harmonic Series Decay
            exploration_rate = self.min_exploration_rate + (
                        self.initial_exploration_rate - self.min_exploration_rate) / (
                                           1 + np.sum(1 / np.arange(1, episode + 2)))
        elif self.current_decay_function == 19:  # Piecewise Linear Decay
            if episode < self.max_episodes / 2:
                exploration_rate = self.initial_exploration_rate - (
                            self.initial_exploration_rate - self.min_exploration_rate) * (
                                               2 * episode / self.max_episodes)
            else:
                exploration_rate = self.min_exploration_rate
        elif self.current_decay_function == 20:  # Custom Polynomial Decay
            p = 3  # Change the power for different polynomial behaviors
            exploration_rate = self.initial_exploration_rate * (1 - (episode / self.max_episodes) ** p)
        else:
            raise ValueError("Invalid decay function number")

        return exploration_rate

class QLearningAgent:
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None, csv_path=None):
        # Load Shared Config
        self.shared_config = load_config(shared_config_path)

        # Load Agent Specific Config if path provided
        if agent_config_path:
            self.agent_config = load_config(agent_config_path)
        else:
            self.agent_config = {}

        # If override_config is provided, merge it with the loaded agent_config
        if override_config:
            self.agent_config.update(override_config)

        # Access the results directory from the shared_config
        self.results_directory = self.shared_config['directories']['results_directory']

        # Create a unique subdirectory for each run to avoid overwriting results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_subdirectory = os.path.join(self.results_directory, "q_learning", run_name, timestamp)
        os.makedirs(self.results_subdirectory, exist_ok=True)

        # Set up logging to the correct directory
        log_file_path = os.path.join(self.results_subdirectory, 'agent_log.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO)
        # Initialize agent-specific configurations and variables
        self.env = env
        self.run_name = run_name
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.learning_rate = self.agent_config['agent']['learning_rate']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']

        # Parameters for adjusting learning rate over time
        self.learning_rate_decay = self.agent_config['agent']['learning_rate_decay']
        self.min_learning_rate = self.agent_config['agent']['min_learning_rate']

        # Initialize q table
        rows = np.prod(env.observation_space.nvec)
        columns = np.prod(env.action_space.nvec)
        self.q_table = np.zeros((rows, columns))

        # Initialize other required variables and structures
        self.training_data = []
        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.possible_states = [list(range(0, (k))) for k in self.env.observation_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]
        self.all_states = [str(i) for i in list(itertools.product(*self.possible_states))]

        self.states = list(itertools.product(*self.possible_states))

        # Initialize state visit counts for count-based exploration
        self.state_visits = np.zeros(rows)

        # moving average for early stopping criteria
        self.moving_average_window = 100  # Number of episodes to consider for moving average
        self.stopping_criterion = 0.01  # Threshold for stopping
        self.prev_moving_avg = -float('inf')  # Initialize to negative infinity to ensure any reward is considered an improvement in the first episode.
        self.state_action_visits = np.zeros((rows, columns))

        self.decay_handler = ExplorationRateDecay(self.max_episodes, self.min_exploration_rate, self.exploration_rate)
        self.decay_function = self.agent_config['agent']['e_decay_function']

        # CSV file for metrics
        self.csv_file_path = os.path.join(self.results_subdirectory, 'training_metrics.csv')

        # Handle CSV input
        if csv_path:
            self.community_risk_values = self.read_community_risk_from_csv(csv_path)
            self.max_weeks = len(self.community_risk_values)
        else:
            self.community_risk_values = None
            self.max_weeks = self.env.campus_state.model.max_weeks

    def read_community_risk_from_csv(self, csv_path):
        try:
            community_risk_df = pd.read_csv(csv_path)
            return community_risk_df['Risk-Level'].tolist()
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
    def log_all_states_visualizations(self, q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory):
        file_paths = visualize_all_states(q_table, all_states, states, run_name, max_episodes, alpha,
                                          results_subdirectory, self.env.students_per_course)

        # Log all generated visualizations
        # wandb_images = [wandb.Image(path) for path in file_paths]
        # wandb.log({"All States Visualization": wandb_images})

        # Log them individually with dimension information
        # for path in file_paths:
        #     infected_dim = path.split('infected_dim_')[-1].split('.')[0]
        #     wandb.log({f"All States Visualization (Infected Dim {infected_dim})": wandb.Image(path)})

    def log_states_visited(self, states, visit_counts, alpha, results_subdirectory):
        file_paths = states_visited_viz(states, visit_counts, alpha, results_subdirectory)

        # Log all generated heatmaps
        # wandb_images = [wandb.Image(path) for path in file_paths]
        # wandb.log({"States Visited": wandb_images})

        # Log them individually with dimension information
        # for path in file_paths:
        #     if "error" in path:
        #         wandb.log({"States Visited Error": wandb.Image(path)})
        #     else:
        #         dim = path.split('infected_dim_')[-1].split('.')[0]
        #         wandb.log({f"States Visited (Infected Dim {dim})": wandb.Image(path)})

    def visualize_q_table(self):
        # Create a heatmap for the Q-table
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.q_table, annot=True, cmap="YlGnBu")
        plt.title("Q-table Heatmap")
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.savefig(os.path.join(self.results_subdirectory, 'q_table_heatmap.png'))
        plt.close()

    def initialize_q_table_from_csv(self, csv_file):
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Loop through the CSV and populate the Q-table
        for index, row in df.iterrows():
            self.q_table[index, 0] = row['Reward 0']
            self.q_table[index, 1] = row['Reward 50']
            self.q_table[index, 2] = row['Reward 100']


    def save_q_table(self):
        policy_dir = self.shared_config['directories']['policy_directory']
        if not os.path.exists(policy_dir):
            os.makedirs(policy_dir)

        file_path = os.path.join(policy_dir, f'q_table_{self.run_name}.npy')
        np.save(file_path, self.q_table)
        print(f"Q-table saved to {file_path}")

    def _policy(self, mode, state):
        state_idx = self.all_states.index(str(tuple(state)))
        if mode == 'train':
            if random.uniform(0, 1) > self.exploration_rate:
                action = np.argmax(self.q_table[state_idx])
            else:
                action = random.randint(0, self.q_table.shape[1] - 1)
        elif mode == 'test':
            action = np.argmax(self.q_table[state_idx])

        # Convert the single action index to a list of actions for each course
        num_courses = len(self.env.action_space.nvec)
        course_actions = self.index_to_action_list(action, self.env.action_space.nvec)

        # Scale the actions to ensure they map to valid indices
        scaled_course_actions = [self.scale_action(a, n) for a, n in zip(course_actions, self.env.action_space.nvec)]

        return scaled_course_actions

    def discretize_state(self, state):
        """
        Discretizes the state to match the format of states in self.all_states.

        Parameters:
        state (list or array): The state to be discretized. This should be a list or array of continuous values.

        Returns:
        list: The discretized state, where each value has been converted into a discrete bin.
        """
        # Discretize each value in the state using the get_discrete_value function
        discrete_state = [self.get_discrete_value(value) for value in state]

        return discrete_state

    def scale_action(self, action, num_actions):
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than 1 to scale actions.")
        max_value = 100
        step_size = max_value / (num_actions - 1)
        return int(action * step_size)

    def action_list_to_index(self, action_list, action_spaces):
        """
        Convert a list of actions into a single action index.
        """
        index = 0
        multiplier = 1
        for a, space in zip(reversed(action_list), reversed(action_spaces)):
            index += a * multiplier
            multiplier *= space
        return int(index)

    def index_to_action_list(self, index, action_spaces):
        action_list = []
        for space in reversed(action_spaces):
            action_list.append(index % space)
            index //= space
        return list(reversed(action_list))

    def reverse_scale_action(self, scaled_action, num_actions):
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than 1 to reverse scale actions.")
        max_value = 100
        step_size = max_value / (num_actions - 1)
        return round(scaled_action / step_size)

    def train(self, alpha):
        """Train the agent."""
        start_time = time.time()
        actual_rewards = []
        predicted_rewards = []
        rewards_per_episode = []
        last_episode = {}
        visited_state_counts = {}
        td_errors = []
        training_log = []
        cumulative_rewards = []
        q_value_diffs = []
        # Initialize accumulators for allowed and infected
        allowed_means_per_episode = []
        infected_means_per_episode = []
        total_steps = 0  # To calculate mean

        c_file_name = f'training_metrics_q_{self.run_name}.csv'
        csv_file_path = os.path.join(self.results_subdirectory, c_file_name)
        file_exists = os.path.isfile(csv_file_path)
        csvfile = open(csv_file_path, 'a', newline='')
        writer = csv.DictWriter(csvfile,
                                fieldnames=['episode', 'cumulative_reward', 'average_reward', 'discounted_reward',
                                            'q_value_change', 'sample_efficiency', 'policy_entropy',
                                            'space_complexity'])
        if not file_exists:
            writer.writeheader()

        previous_q_table = np.copy(self.q_table)

        for episode in tqdm(range(self.max_episodes)):
            self.decay_handler.set_decay_function(self.decay_function)
            state = self.env.reset()
            c_state = state[0]
            terminated = False
            e_return = []
            total_reward = 0
            step = 0
            episode_td_errors = []
            last_action = None
            policy_changes = 0
            episode_visited_states = set()
            q_values_list = []
            episode_allowed = []
            episode_infected = []

            while not terminated:
                action = self._policy('train', c_state)
                converted_state = str(tuple(c_state))
                state_idx = self.all_states.index(converted_state)
                episode_visited_states.add(converted_state)

                # action is already a list of scaled actions, so we don't need to scale it again
                action_alpha_list = [*action, alpha]


                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                next_state = self.discretize_state(next_state)

                # Convert the scaled action back to an index for Q-table update
                action_idx = self.action_list_to_index(
                    [self.reverse_scale_action(a, n) for a, n in zip(action, self.env.action_space.nvec)],
                    self.env.action_space.nvec)

                if action_idx >= self.q_table.shape[1]:
                    raise IndexError(
                        f"Action index {action_idx} is out of bounds for Q-table with shape {self.q_table.shape}")

                try:
                    old_value = self.q_table[state_idx, action_idx]
                except IndexError as e:
                    logging.error(f"IndexError in Q-table access: {e}")
                    raise

                next_max = np.max(self.q_table[self.all_states.index(str(tuple(next_state)))])

                # Update the total allowed and infected counts
                episode_allowed.append(sum(info.get('allowed', [])))  # Sum all courses' allowed students
                episode_infected.append(sum(info.get('infected', [])))

                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                        reward + self.discount_factor * next_max)
                self.q_table[state_idx, action_idx] = new_value

                td_error = abs(reward + self.discount_factor * next_max - old_value)
                episode_td_errors.append(td_error)

                if last_action is not None and last_action != action_idx:
                    policy_changes += 1
                last_action = action_idx

                q_values_list.append(self.q_table[state_idx])

                self.state_action_visits[state_idx, action_idx] += 1
                self.state_visits[state_idx] += 1

                total_reward += reward
                e_return.append(reward)

                step += 1
                c_state = next_state

                # print(info)

            avg_episode_return = sum(e_return) / len(e_return)
            cumulative_rewards.append(total_reward)
            rewards_per_episode.append(avg_episode_return)
            avg_td_error = np.mean(episode_td_errors)
            td_errors.append(avg_td_error)

            unique_state_visits = len(episode_visited_states)
            sample_efficiency = unique_state_visits

            # Log visited states and their counts
            for state in episode_visited_states:
                visited_state_counts[state] = visited_state_counts.get(state, 0) + 1

            q_values = np.array(q_values_list)
            exp_q_values = np.exp(
                q_values - np.max(q_values, axis=1, keepdims=True))
            probabilities = exp_q_values / np.sum(exp_q_values, axis=1, keepdims=True)
            policy_entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1).mean()

            q_value_change = np.mean((self.q_table - previous_q_table) ** 2)
            q_value_diffs.append(q_value_change)
            previous_q_table = np.copy(self.q_table)

            metrics = {
                'episode': episode,
                'cumulative_reward': total_reward,
                'average_reward': avg_episode_return,
                'discounted_reward': sum([r * (self.discount_factor ** i) for i, r in enumerate(e_return)]),
                'q_value_change': q_value_change,
                'sample_efficiency': sample_efficiency,
                'policy_entropy': policy_entropy,
                'space_complexity': self.q_table.nbytes
            }
            writer.writerow(metrics)
            # wandb.log({'cumulative_reward': total_reward})
            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)
            e_mean_allowed = sum(episode_allowed) / len(episode_allowed)
            e_mean_infected = sum(episode_infected) / len(episode_infected)
            allowed_means_per_episode.append(e_mean_allowed)
            infected_means_per_episode.append(e_mean_infected)
            # Increment the episode count


        csvfile.close()
        print("Training complete.")
        # Calculate the means for allowed and infected
        mean_allowed = round(sum(allowed_means_per_episode) / len(allowed_means_per_episode))
        mean_infected = round(sum(infected_means_per_episode) / len(infected_means_per_episode))


        # Save the results in a separate CSV file
        summary_file_path = os.path.join(self.results_subdirectory, 'mean_allowed_infected.csv')
        with open(summary_file_path, 'w', newline='') as summary_csvfile:
            summary_writer = csv.DictWriter(summary_csvfile, fieldnames=['mean_allowed', 'mean_infected'])
            summary_writer.writeheader()
            summary_writer.writerow({'mean_allowed': mean_allowed, 'mean_infected': mean_infected})
        self.save_q_table()

        self.save_training_log_to_csv(training_log)

        visualize_q_table(self.q_table, self.results_subdirectory, self.max_episodes)

        # Convert visited_state_counts dictionary to lists for logging
        states = list(visited_state_counts.keys())
        visit_counts = list(visited_state_counts.values())
        self.log_states_visited(states, visit_counts, alpha, self.results_subdirectory)
        self.log_all_states_visualizations(self.q_table, self.all_states, self.states, self.run_name, self.max_episodes,
                                           alpha, self.results_subdirectory)

        return actual_rewards

    def save_training_log_to_csv(self, training_log, init_method='default-1'):
        # Define the CSV file path
        csv_file_path = os.path.join(self.results_subdirectory, f'training_log_{init_method}.csv')

        # Write the training log to the CSV file
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write headers
            writer.writerow(
                ['Episode', 'Step', 'Total Reward', 'Average TD Error', 'Policy Changes', 'Exploration Rate'])
            # Write training log data
            writer.writerows(training_log)

        print(f"Training log saved to {csv_file_path}")

    def identify_safety_set(self, allowed_values_over_time, infected_values_over_time, x, y, z,
                            evaluation_subdirectory):
        """Identify and plot the safety set based on fixed constraints."""

        infected_values_over_time = [val[0] if isinstance(val, (list, tuple)) else val for val in
                                     infected_values_over_time]

        # 1. Ensure that no more than `x` infected individuals are present for more than `z%` of the time.
        time_exceeding_x = sum(1 for val in infected_values_over_time if val > x)
        time_within_x = len(infected_values_over_time) - time_exceeding_x
        infection_safety_percentage = (time_within_x / len(infected_values_over_time)) * 100

        # 2. Ensure that `y` allowed students are present at least `z%` of the time.
        time_with_y_present = sum(1 for val in allowed_values_over_time if val >= y)
        attendance_safety_percentage = (time_with_y_present / len(allowed_values_over_time)) * 100

        with open(os.path.join(evaluation_subdirectory, 'safety_set.txt'), 'w') as f:
            f.write(f"Safety Condition: No more than {x} infections for {100 - z}% of time: "
                    f"{infection_safety_percentage}%\n")
            f.write(f"Safety Condition: At least {y} allowed students for {z}% of time: "
                    f"{attendance_safety_percentage}%\n")

        plt.figure(figsize=(10, 6))
        plt.scatter(allowed_values_over_time, infected_values_over_time, color='blue', label='State Points')
        plt.axhline(y=x, color='red', linestyle='--', label=f'Infection Threshold (x={x})')
        plt.axvline(x=y, color='green', linestyle='--', label=f'Attendance Threshold (y={y})')
        plt.xlabel('Allowed Students')
        plt.ylabel('Infected Individuals')
        plt.legend()
        plt.title('Safety Set Identification')
        plt.grid(True)
        plt.savefig(os.path.join(evaluation_subdirectory, 'safety_set.png'))
        plt.close()

        # Check if the conditions are met
        infection_condition_met = infection_safety_percentage >= (100 - z)
        attendance_condition_met = attendance_safety_percentage >= z

        return infection_condition_met, attendance_condition_met

    def construct_cbf(self, allowed_values_over_time, infected_values_over_time, evaluation_subdirectory, x, y):
        """Construct and save the Control Barrier Function (CBF) based on fixed safety constraints."""

        processed_infected_values = [
            infected[0] if isinstance(infected, (list, tuple)) else infected
            for infected in infected_values_over_time
        ]

        # Directly use the provided x and y values
        B1 = lambda s: x - (s[1][0] if isinstance(s[1], (list, tuple)) else s[1])
        B2 = lambda s: (s[0] - y)

        with open(os.path.join(evaluation_subdirectory, 'cbf.txt'), 'w') as f:
            f.write(f"CBF for Infections: B1(s) = {x} - Infected Individuals\n")
            f.write(f"CBF for Attendance: B2(s) = Allowed Students - {y}\n")

        return B1, B2

    def verify_forward_invariance(self, B1, B2, allowed_values_over_time, infected_values_over_time,
                                  evaluation_subdirectory):
        """Verify forward invariance using the constructed CBFs."""
        is_invariant = True
        for i in range(len(allowed_values_over_time) - 1):
            s_t = [allowed_values_over_time[i], infected_values_over_time[i]]
            s_t_plus_1 = [allowed_values_over_time[i + 1], infected_values_over_time[i + 1]]

            # Calculate derivatives with correct unpacking
            dB1_dt = B1(s_t_plus_1) - B1(s_t)
            dB2_dt = B2(s_t_plus_1) - B2(s_t)

            if not (dB1_dt >= 0 and dB2_dt >= 0):
                is_invariant = False
                break

        # Save the verification result
        with open(os.path.join(evaluation_subdirectory, 'cbf_verification.txt'), 'w') as f:
            if is_invariant:
                f.write("The forward invariance of the system is verified using the constructed CBFs.\n")
            else:
                f.write("The system is not forward invariant based on the constructed CBFs.\n")

        return is_invariant

    def extract_features(self, allowed_values_over_time, infected_values_over_time):
        """Extract features for constructing the Lyapunov function."""
        features = []
        for i in range(len(allowed_values_over_time)):
            features.append([allowed_values_over_time[i], infected_values_over_time[i]])
        return features

    def flatten_features(self, feature):
        if isinstance(feature, (list, tuple)):
            flattened_feature = [item for sublist in feature for item in
                                 (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        else:
            flattened_feature = [feature]  # Handle the case where feature is a single scalar

        # Ensure the flattened feature is a 3-element list for use in the neural network
        if len(flattened_feature) > 3:
            flattened_feature = flattened_feature[:3]  # Truncate if there are more than 3 elements
        elif len(flattened_feature) < 3:
            flattened_feature.extend(
                [0] * (3 - len(flattened_feature)))  # Pad with zeros if there are fewer than 3 elements

        return flattened_feature

    # Function to construct the Lyapunov function
    def construct_lyapunov_function(self, features, alpha):
        model = LyapunovNet(input_dim=2, hidden_dim=64, output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_values = []
        epochs = 1000
        epsilon = 1e-6

        # Generate diverse states for training
        train_states = self.generate_diverse_states(1000)
        train_states = train_states.float()

        for epoch in range(epochs):
            optimizer.zero_grad()
            V = model(train_states)
            V_next = model(self.get_next_states(train_states, alpha).float())

            positive_definite_loss = F.relu(-V + epsilon).mean()
            decreasing_loss = F.relu(V_next - V + epsilon).mean()
            loss = positive_definite_loss + decreasing_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf detected at epoch {epoch}")
                continue

            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())

        torch.save(model.state_dict(), os.path.join(self.results_subdirectory, 'lyapunov_model.pth'))

        return model, loss_values

    def generate_diverse_states(self, num_samples):
        infected = np.random.uniform(0, 100, num_samples)
        risk = np.random.uniform(0, 1, num_samples)
        return torch.tensor(np.column_stack((infected, risk)), dtype=torch.float32)

    def plot_loss_function(self, loss_values, alpha, run_name):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Lyapunov Function Training Loss (Run: {run_name}, Alpha: {alpha})')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_subdirectory, f'lyapunov_loss_plot_{run_name}_alpha_{alpha}.png'))
        plt.close()

    def plot_steady_state_and_stable_points(self, V, features, run_name, alpha):
        infected = np.linspace(0, 100, 100)
        risk = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(infected, risk)
        states = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)

        with torch.no_grad():
            V_values = V(states).squeeze().numpy().reshape(X.shape)
            steady_state = np.exp(-V_values / V_values.max())
            steady_state /= steady_state.sum()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        c1 = ax1.contourf(X, Y, steady_state, levels=20, cmap='viridis')
        ax1.set_title(f'Steady-State Distribution (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('Infected')
        ax1.set_ylabel('Community Risk')
        cbar1 = fig.colorbar(c1, ax=ax1)
        cbar1.set_label('Steady-State Probability')

        c2 = ax2.contourf(X, Y, V_values, levels=20, cmap='coolwarm')
        ax2.set_title(f'Lyapunov Function (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('Infected')
        ax2.set_ylabel('Community Risk')
        cbar2 = fig.colorbar(c2, ax=ax2)
        cbar2.set_label('Lyapunov Function Value')

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_subdirectory, f'steady_state_and_stable_points_{run_name}_alpha_{alpha}.png'))
        plt.close()

    # Plotting the Lyapunov function and changes
    def plot_lyapunov_change(self, V, features, run_name, alpha):
        infected = np.linspace(0, 100, 50)
        risk = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(infected, risk)
        states = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)

        with torch.no_grad():
            V_values = V(states).squeeze().numpy().reshape(X.shape)
            next_states = self.get_next_states(states, alpha)
            V_next_values = V(next_states.float()).squeeze().numpy().reshape(X.shape)
            delta_V = V_next_values - V_values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        im1 = ax1.imshow(V_values, extent=[0, 100, 0, 1], origin='lower', aspect='auto', cmap='viridis')
        ax1.set_title(f'Lyapunov Function V(x) (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('Infected')
        ax1.set_ylabel('Community Risk')
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar1.set_label('V(x)')

        im2 = ax2.imshow(delta_V, extent=[0, 100, 0, 1], origin='lower', aspect='auto', cmap='coolwarm')
        ax2.set_title(f'Change in Lyapunov Function ΔV(x) (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('Infected')
        ax2.set_ylabel('Community Risk')
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label('ΔV(x)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_subdirectory, f'lyapunov_change_{run_name}_alpha_{alpha}.png'))
        plt.close()

    def plot_equilibrium_points(self, features, run_name, alpha):
        infected = [f[0] for f in features]
        risk = [f[1] for f in features]
        plt.figure(figsize=(10, 8))
        plt.scatter(risk, infected, c=infected, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Number of Infected')
        plt.xlabel('Community Risk')
        plt.ylabel('Number of Infected')
        plt.title(f'Equilibrium Points: DFE and EE Regions (Run: {run_name}, Alpha: {alpha})')
        plt.axhline(y=0.5, color='r', linestyle='--', label='DFE/EE Boundary')
        plt.text(0.5, 0.25, 'DFE Region', horizontalalignment='center')
        plt.text(0.5, 0.75, 'EE Region', horizontalalignment='center')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_subdirectory, f'equilibrium_points_{run_name}_alpha_{alpha}.png'))
        plt.close()

    def plot_lyapunov_properties(self, V, features, run_name, alpha):
        eval_states = torch.tensor([[f[0], f[1]] for f in features], dtype=torch.float32)

        with torch.no_grad():
            V_values = V(eval_states).squeeze()
            next_states = self.get_next_states(eval_states, alpha)
            V_next = V(next_states).squeeze()
            delta_V = V_next - V_values

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

        # Plot V(x)
        ax1.plot(V_values.numpy(), label='V(x)')
        ax1.axhline(y=0, color='r', linestyle='--', label='V(x) = 0')
        ax1.set_title(f'Lyapunov Function Values (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('State Index')
        ax1.set_ylabel('V(x)')
        ax1.legend()
        ax1.grid(True)

        # Plot ΔV(x)
        ax2.plot(delta_V.numpy(), label='ΔV(x)')
        ax2.axhline(y=0, color='r', linestyle='--', label='ΔV(x) = 0')
        ax2.set_title(f'Change in Lyapunov Function (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('State Index')
        ax2.set_ylabel('ΔV(x)')
        ax2.legend()
        ax2.grid(True)

        # Plot V(x) vs ΔV(x)
        ax3.scatter(V_values.numpy(), delta_V.numpy(), alpha=0.6)
        ax3.axhline(y=0, color='r', linestyle='--', label='ΔV(x) = 0')
        ax3.axvline(x=0, color='g', linestyle='--', label='V(x) = 0')
        ax3.set_title(f'V(x) vs ΔV(x) (Run: {run_name}, Alpha: {alpha})')
        ax3.set_xlabel('V(x)')
        ax3.set_ylabel('ΔV(x)')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_subdirectory, f'lyapunov_properties_{run_name}_alpha_{alpha}.png'))
        plt.close()

        positive_definite = (V_values > 0).float().mean()
        decreasing = (delta_V < 0).float().mean()

        print(f"Positive definite: {positive_definite.item():.2%}")
        print(f"Decreasing: {decreasing.item():.2%}")

        return positive_definite.item(), decreasing.item()

    def get_discrete_value(self, number):
        """
        Converts a given number to a discrete value based on its range.

        Parameters:
        number (int or float): The input number to be converted to a discrete value.

        Returns:
        int: A discrete value representing the range in which the input number falls.
             It returns a value between 0 and 9, inclusive.

        Example:
        get_discrete_value(25) returns 2
        get_discrete_value(99) returns 9
        """

        # Ensure the number is within the range [0, 100]
        number = min(99, max(0, number))

        # Perform integer division by 10 to get the discrete value
        # This will also ensure that the returned value is an integer
        return number // 10

    def get_next_states(self, states, alpha):
        next_states = []
        for state in states:
            current_infected = int(state[0].item())
            community_risk = int(state[1].item() * 100)
            discrete_state = [self.get_discrete_value(current_infected), self.get_discrete_value(community_risk)]
            state_str = str(tuple(discrete_state))
            if state_str not in self.all_states:
                print(f"Error: Generated state {discrete_state} is not in self.all_states")
                print(f"All possible states: {self.all_states}")
                raise ValueError(f"State {discrete_state} not found in self.all_states")
            state_idx = self.all_states.index(str(tuple(discrete_state)))
            action = np.argmax(self.q_table[state_idx])
            action_value = action * 50
            alpha_infection = 0.005
            beta_infection = 0.01
            new_infected = int(((alpha_infection * current_infected) * action_value) + (
                        (beta_infection * community_risk / 100) * action_value ** 2))
            new_infected = min(new_infected, action_value)
            next_states.append(torch.tensor([new_infected, community_risk / 100], dtype=torch.float32))

        return torch.stack(next_states)

    def verify_lyapunov_stability(self, V, theta, features, evaluation_subdirectory):
        """Verify the stochastic stability using the Lyapunov function in both DFE and EE regions."""
        dfe_stable = True
        ee_stable = True
        dfe_points = 0
        ee_points = 0
        dfe_stable_points = 0
        ee_stable_points = 0

        def flatten_feature(feature):
            """Flatten the feature list if it's nested and ensure it has 3 elements."""
            flattened_feature = [item for sublist in feature for item in
                                 (sublist if isinstance(sublist, list) else [sublist])]
            if len(flattened_feature) > 3:
                flattened_feature = flattened_feature[:3]
            elif len(flattened_feature) < 3:
                flattened_feature.extend([0] * (3 - len(flattened_feature)))
            return flattened_feature

        for i in range(len(features) - 1):
            flattened_feature = flatten_feature(features[i])
            flattened_next_feature = flatten_feature(features[i + 1])

            feature_tensor = torch.tensor(flattened_feature, dtype=torch.float32)
            next_feature_tensor = torch.tensor(flattened_next_feature, dtype=torch.float32)

            try:
                V_t = V(feature_tensor)
                V_t_plus_1 = V(next_feature_tensor)
                delta_V = V_t_plus_1 - V_t

                if features[i][1] == 0:  # DFE region
                    dfe_points += 1
                    if delta_V.item() <= 0:
                        dfe_stable_points += 1
                    else:
                        dfe_stable = False
                else:  # EE region
                    ee_points += 1
                    if delta_V.item() <= 0:
                        ee_stable_points += 1
                    else:
                        ee_stable = False

            except Exception as e:
                print(f"Error during V function evaluation: {e}")
                raise

        with open(os.path.join(evaluation_subdirectory, 'lyapunov_verification.txt'), 'w') as f:
            if dfe_points == 0:
                f.write(
                    "There is insufficient data to determine stability in the Disease-Free Equilibrium (DFE) region.\n")
                dfe_stable = False  # Set to False if no data
            else:
                dfe_stability_percentage = (dfe_stable_points / dfe_points) * 100
                f.write(
                    f"The system is {dfe_stability_percentage:.2f}% stable in the Disease-Free Equilibrium (DFE) region.\n")
                f.write("The system is " + ("stochastically stable" if dfe_stable else "not stochastically stable") +
                        " in the Disease-Free Equilibrium (DFE) region based on the Lyapunov function.\n")

            if ee_points == 0:
                f.write("There is no data for the Endemic Equilibrium (EE) region.\n")
                ee_stable = False  # Set to False if no data
            else:
                ee_stability_percentage = (ee_stable_points / ee_points) * 100
                f.write(
                    f"The system is {ee_stability_percentage:.2f}% stable in the Endemic Equilibrium (EE) region.\n")
                f.write("The system is " + ("stochastically stable" if ee_stable else "not stochastically stable") +
                        " in the Endemic Equilibrium (EE) region based on the Lyapunov function.\n")

        return dfe_stable, ee_stable

    import numpy as np

    def calculate_stationary_distribution(self, states, next_states):
        unique_states = sorted(set([tuple(state) for state in states + next_states]))
        state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
        num_states = len(unique_states)

        # Initialize the transition matrix
        P = np.zeros((num_states, num_states))

        # Fill the transition matrix
        for current_state, next_state in zip(states, next_states):
            i = state_to_idx[tuple(current_state)]
            j = state_to_idx[tuple(next_state)]
            P[i, j] += 1

        # Normalize rows to get probabilities
        row_sums = P.sum(axis=1, keepdims=True)

        # Handle rows with zero sum to prevent division by zero
        for i, row_sum in enumerate(row_sums):
            if row_sum == 0:
                # Assign a small probability to all transitions (smoothing)
                P[i, :] = 1.0 / num_states
            else:
                P[i, :] /= row_sum



        # Solve for the stationary distribution
        eigvals, eigvecs = np.linalg.eig(P.T)

        # Identify the eigenvectors corresponding to an eigenvalue of 1
        close_to_one = np.isclose(eigvals, 1)

        if not np.any(close_to_one):
            raise ValueError("No eigenvalues close to 1 found; unable to compute stationary distribution.")

        stationary_distribution = eigvecs[:, close_to_one]

        if stationary_distribution.size == 0:
            raise ValueError(
                "Stationary distribution is empty. This might indicate a problem with the transition matrix.")

        stationary_distribution = stationary_distribution[:, 0].real
        stationary_distribution = stationary_distribution / stationary_distribution.sum()

        return unique_states, stationary_distribution

    def plot_equilibrium_points_with_stationary_distribution(self, stationary_distribution, unique_states, run_name):
        infected = [state[0] for state in unique_states]
        risk = [state[1] for state in unique_states]

        plt.figure(figsize=(10, 8))
        plt.scatter(risk, infected, c=stationary_distribution, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Stationary Distribution')
        plt.xlabel('Community Risk')
        plt.ylabel('Number of Infected')
        plt.title(f'Equilibrium Points with Stationary Distribution (Run: {run_name})')
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_subdirectory, f'equilibrium_points_stationary_distribution_{run_name}.png'))
        plt.close()

    def plot_transition_matrix_using_risk(self, states, next_states, community_risks, run_name,
                                          evaluation_subdirectory):
        states = [tuple(state) for state in states]
        next_states = [tuple(next_state) for next_state in next_states]
        unique_states = sorted(set(states + next_states))
        state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
        num_states = len(unique_states)

        P = np.zeros((num_states, num_states))
        for state, next_state, risk in zip(states, next_states, community_risks):
            transition_probability = risk
            P[state_to_idx[state], state_to_idx[next_state]] += transition_probability

        P = P / P.sum(axis=1, keepdims=True)
        P_log = np.log10(P + 1e-10)

        plt.figure(figsize=(10, 8))
        plt.imshow(P_log, cmap='gray', aspect='auto', origin='lower')
        plt.colorbar(label='Log10 of Transition Probability')
        plt.xlabel('Next State Index')
        plt.ylabel('Current State Index')
        plt.title(f'Transition Probability Matrix (Log Scale)\nRun: {run_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_subdirectory, f'transition_matrix_{run_name}.png'))
        plt.close()

    def plot_evaluation_results(self, allowed_values_over_time, infected_values_over_time, community_risk_values,
                                run_name, evaluation_subdirectory):
        plt.clf()
        plt.close('all')
        print("Plotting evaluation results...")

        sns.set(style="whitegrid")

        # Create a figure and a set of subplots
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Bar plot for allowed and infected values on primary y-axis
        bar_width = 0.4
        weeks = np.arange(len(allowed_values_over_time))

        ax1.bar(weeks - bar_width / 2, allowed_values_over_time, width=bar_width, color='blue', alpha=0.6,
                label='Allowed')
        ax1.bar(weeks + bar_width / 2, infected_values_over_time, width=bar_width, color='red', alpha=0.6,
                label='Infected')

        ax1.set_xlabel('Week')
        ax1.set_ylabel('Allowed and Infected Values')
        ax1.legend(loc='upper left')

        # Create a secondary y-axis for community risk
        ax2 = ax1.twinx()
        sns.lineplot(x=weeks, y=community_risk_values, marker='s', linestyle='--', color='black', linewidth=2.5, ax=ax2)
        ax2.set_ylabel('Community Risk', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Adjust layout and save the plot
        plt.tight_layout()

        os.makedirs(evaluation_subdirectory, exist_ok=True)
        plot_path = os.path.join(evaluation_subdirectory, f"evaluation_plot_{run_name}.png")

        try:
            plt.savefig(plot_path)
            print(f"Evaluation results plot saved to {plot_path}")
        except Exception as e:
            print(f"Error saving evaluation results plot: {e}")
        finally:
            plt.close()

    def evaluate_lyapunov(self, V, eval_states, alpha):
        with torch.no_grad():
            eval_states = eval_states.float()  # Ensure the eval_states are Float32
            V_values = V(eval_states).squeeze()
            next_states = self.get_next_states(eval_states, alpha)
            V_next = V(next_states.float()).squeeze()

            positive_definite = (V_values > 0).float().mean()
            decreasing = (V_next < V_values).float().mean()

        print(f"Positive definite: {positive_definite.item():.2%}")
        print(f"Decreasing: {decreasing.item():.2%}")

        return positive_definite.item(), decreasing.item()

    def find_optimal_xy(self, infected_values, allowed_values, community_risk_values, z=95, run_name=None,
                        evaluation_subdirectory=None):
        # Binary search for smallest x (infection threshold)
        low_x, high_x = 0, max(infected_values)
        optimal_x = high_x

        while low_x <= high_x:
            mid_x = (low_x + high_x) // 2
            time_lessthan_mid_x = sum(1 for val in infected_values if val < mid_x)
            infection_safety_percentage = ((len(infected_values) - time_lessthan_mid_x) / len(infected_values)) * 100

            if infection_safety_percentage >= z:
                optimal_x = mid_x
                high_x = mid_x - 1
            else:
                low_x = mid_x + 1

        # Binary search for largest y (allowed threshold)
        low_y, high_y = 0, max(allowed_values)
        optimal_y = low_y

        while low_y <= high_y:
            mid_y = (low_y + high_y) // 2
            time_with_mid_y_present = sum(1 for val in allowed_values if val >= mid_y)
            attendance_safety_percentage = (time_with_mid_y_present / len(allowed_values)) * 100

            if attendance_safety_percentage >= z:
                optimal_y = mid_y
                low_y = mid_y + 1
            else:
                high_y = mid_y - 1

        # Plotting the safety conditions
        if run_name and evaluation_subdirectory:
            plt.figure(figsize=(10, 6))
            plt.scatter(allowed_values, infected_values, color='blue', label='State Points')
            plt.axhline(y=optimal_x, color='red', linestyle='--', label=f'Infection Threshold (x={optimal_x})')
            plt.axvline(x=optimal_y, color='green', linestyle='--', label=f'Attendance Threshold (y={optimal_y})')
            plt.xlabel('Allowed Students')
            plt.ylabel('Infected Individuals')
            plt.legend()
            plt.title(f'Safety Set Identification - {run_name}')
            plt.grid(True)
            plt.savefig(os.path.join(evaluation_subdirectory, f'safety_set_plot_{run_name}.png'))
            plt.close()

        return optimal_x, optimal_y

    def simulate_steady_state(self, num_simulations=10000, num_steps=100, alpha=0.5):
        """
        Simulate the system for a long time to estimate the steady-state distribution.

        Args:
        num_simulations (int): Number of independent simulations to run.
        num_steps (int): Number of steps for each simulation.
        alpha (float): The alpha parameter for the policy.

        Returns:
        numpy.ndarray: Array of final states from all simulations.
        """
        final_states = []

        for _ in range(num_simulations):
            infected = np.random.uniform(0, 100)  # Initial infected count
            risk = np.random.uniform(0, 1)  # Initial community risk

            for _ in range(num_steps):
                state = [self.get_discrete_value(infected), self.get_discrete_value(risk)]
                state_idx = self.all_states.index(str(tuple(int(value) for value in state)))


                action = np.argmax(self.q_table[state_idx])
                allowed_value = action * 50  # Convert to allowed values (0, 50, 100)

                alpha_infection = 0.005
                beta_infection = 0.01

                new_infected = ((alpha_infection * infected) * allowed_value) + \
                               ((beta_infection * risk) * allowed_value ** 2)

                infected = min(new_infected, allowed_value)
                risk = np.random.uniform(0, 1)  # Assuming risk is randomly distributed each step

            final_states.append([infected, risk])

        return np.array(final_states)

    def plot_simulated_steady_state(self, final_states, run_name, alpha):
        """
        Plot the simulated steady-state distribution.

        Args:
        final_states (numpy.ndarray): Array of final states from simulations.
        run_name (str): Name of the current run for saving the plot.
        alpha (float): The alpha parameter used in the simulation.
        """
        plt.figure(figsize=(10, 8))

        # Calculate the 2D histogram
        hist, xedges, yedges = np.histogram2d(final_states[:, 1], final_states[:, 0], bins=50,
                                              range=[[0, 1], [0, 100]])

        # Plot the 2D histogram
        plt.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='gray_r')

        plt.colorbar(label='Frequency')
        plt.xlabel('Community Risk')
        plt.ylabel('Infected Individuals')
        plt.title(f'Simulated Steady-State Distribution\nRun: {run_name}, Alpha: {alpha}')

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_subdirectory, f'simulated_steady_state_{run_name}_alpha_{alpha}.png'))
        plt.close()



    def evaluate(self, run_name, num_episodes=1, x_value=38, y_value=80, z=95, alpha=0.5, csv_path=None):
        policy_dir = self.shared_config['directories']['policy_directory']
        q_table_path = os.path.join(policy_dir, f'q_table_{run_name}.npy')
        results_directory = self.results_subdirectory
        infected_values_over_time = [20]

        if not os.path.exists(q_table_path):
            raise FileNotFoundError(f"Q-table file not found in {q_table_path}")

        self.q_table = np.load(q_table_path)
        print(f"Loaded Q-table from {q_table_path}")

        total_rewards = []

        evaluation_subdirectory = os.path.join(results_directory, run_name)
        os.makedirs(evaluation_subdirectory, exist_ok=True)
        csv_file_path = os.path.join(evaluation_subdirectory, f'evaluation_metrics_{run_name}.csv')

        safety_log_path = os.path.join(evaluation_subdirectory, f'safety_conditions_{run_name}.csv')
        interpretation_path = os.path.join(evaluation_subdirectory, f'safety_conditions_interpretation_{run_name}.txt')

        if csv_path:
            self.community_risk_values = self.read_community_risk_from_csv(csv_path)
            self.max_weeks = len(self.community_risk_values)
        else:
            raise ValueError("CSV path for community risk values is required for evaluation.")

        all_allowed_values = []
        all_infected_values = []
        all_community_risk_values = []
        print("Starting evaluation...")

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Step', 'State', 'Action', 'Community Risk', 'Total Reward'])

            with open(safety_log_path, mode='w', newline='') as safety_file, open(interpretation_path,
                                                                                  mode='w') as interpretation_file:
                safety_writer = csv.writer(safety_file)
                safety_writer.writerow(['Episode', 'Infections > 30 (%)', 'Safety Condition Met (Infection)',
                                        'Allowed Students ≥ 100 (%)', 'Safety Condition Met (Attendance)'])

                states = []
                next_states = []
                community_risks = []

                for episode in range(num_episodes):
                    print(f"Starting evaluation for episode {episode + 1}...")
                    state, _ = self.env.reset()
                    print(f"Initial state for episode {episode + 1}: {state}")
                    c_state = state
                    terminated = False
                    total_reward = 0
                    step = 0

                    allowed_values_over_time = []
                    infected_values_over_time = [20]  # Assuming an initial infection count

                    while not terminated:
                        action = self._policy('test', c_state)
                        # c_list_action = [i * 50 for i in action]
                        c_list_action = action  # Pass the arbitrary actions directly

                        action_alpha_list = [*c_list_action, alpha]
                        next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                        next_state = self.discretize_state(next_state)
                        c_state = next_state
                        total_reward += reward

                        states.append(info['continuous_state'])
                        next_states.append(next_state)
                        community_risks.append(info['community_risk'])

                        writer.writerow(
                            [episode + 1, step + 1, info['continuous_state'], c_list_action[0], info['community_risk'],
                             reward]
                        )

                        allowed_values_over_time.append(c_list_action[0])
                        infected_values_over_time.append(
                            info['continuous_state'][0] if isinstance(info['continuous_state'], (list, tuple)) else
                            info['continuous_state'])

                        step += 1

                    total_rewards.append(total_reward)
                    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

                    # Ensure all arrays have the same length
                    min_length = min(len(allowed_values_over_time), len(infected_values_over_time),
                                     len(self.community_risk_values))
                    allowed_values_over_time = allowed_values_over_time[:min_length]
                    infected_values_over_time = infected_values_over_time[:min_length]
                    community_risk_values = self.community_risk_values[:min_length]

                    all_allowed_values.extend(allowed_values_over_time)
                    all_infected_values.extend(infected_values_over_time)
                    all_community_risk_values.extend(community_risk_values)

                    # Calculate the percentage of time infections exceed the threshold
                    time_above_x = sum(1 for val in infected_values_over_time if val > x_value)
                    infection_exceed_percentage = (time_above_x / len(infected_values_over_time)) * 100

                    time_with_y_present = sum(1 for val in allowed_values_over_time if val >= y_value)
                    attendance_safety_percentage = (time_with_y_present / len(allowed_values_over_time)) * 100

                    # Determine if the safety conditions are met
                    infection_condition_met = infection_exceed_percentage <= (100 - z)
                    attendance_condition_met = attendance_safety_percentage >= z

                    # Log the safety condition results in the table
                    safety_writer.writerow([
                        episode + 1,
                        infection_exceed_percentage,
                        'Yes' if infection_condition_met else 'No',
                        attendance_safety_percentage,
                        'Yes' if attendance_condition_met else 'No'
                    ])

                    # Write the interpretation to the text file
                    interpretation_file.write(
                        f"Episode {episode + 1} Interpretation:\n"
                        f"Safety Condition: No more than {x_value} infections for {100 - z}% of time: "
                        f"{infection_exceed_percentage:.2f}% -> {'Condition Met' if infection_condition_met else 'Condition Not Met'}\n"
                        f"Safety Condition: At least {y_value} allowed students for {z}% of time: "
                        f"{attendance_safety_percentage:.2f}% -> {'Condition Met' if attendance_condition_met else 'Condition Not Met'}\n\n"
                    )

                    # Plotting safety conditions
                    plt.figure(figsize=(10, 6))
                    plt.scatter(allowed_values_over_time, infected_values_over_time, color='blue', label='State Points')
                    plt.axhline(y=x_value, color='red', linestyle='--', label=f'Infection Threshold (x={x_value})')
                    plt.axvline(x=y_value, color='green', linestyle='--', label=f'Attendance Threshold (y={y_value})')
                    plt.xlabel('Allowed Students')
                    plt.ylabel('Infected Individuals')
                    plt.legend()
                    plt.title(f'Safety Set Identification - Episode {episode + 1}')
                    plt.grid(True)
                    plt.savefig(os.path.join(evaluation_subdirectory, f'safety_set_plot_episode_{run_name}.png'))
                    plt.close()

                    # Construct CBFs using direct x and y values
                    B1, B2 = self.construct_cbf(allowed_values_over_time, infected_values_over_time,
                                                evaluation_subdirectory, x_value, y_value)

                    # Verify forward invariance
                    is_invariant = self.verify_forward_invariance(B1, B2, allowed_values_over_time,
                                                                  infected_values_over_time, evaluation_subdirectory)

                    # After the episode loop
                    # After the episode loop
                    features = list(zip(allowed_values_over_time, infected_values_over_time,
                                        self.community_risk_values[:len(allowed_values_over_time)]))
                    V, loss_values = self.construct_lyapunov_function(features, alpha)

                    # Evaluate Lyapunov function using the CSV data
                    eval_states = torch.tensor([[f[0], f[1]] for f in features], dtype=torch.float32)
                    self.evaluate_lyapunov(V, eval_states, alpha)

                    self.plot_loss_function(loss_values, alpha, run_name)
                    self.plot_steady_state_and_stable_points(V, features, run_name, alpha)
                    self.plot_lyapunov_change(V, features, run_name, alpha)
                    self.plot_equilibrium_points(features, run_name, alpha)
                    self.plot_lyapunov_properties(V, features, run_name, alpha)

                    # Calculate the stationary distribution and unique states
                    unique_states, stationary_distribution = self.calculate_stationary_distribution(states, next_states)

                    # Plot the equilibrium points with the stationary distribution
                    self.plot_equilibrium_points_with_stationary_distribution(stationary_distribution, unique_states,
                                                                              run_name)

                # After all episodes have been evaluated
                self.plot_transition_matrix_using_risk(states, next_states, community_risks, run_name,
                                                       evaluation_subdirectory)
                optimal_x, optimal_y = self.find_optimal_xy(infected_values_over_time, allowed_values_over_time,
                                                            self.community_risk_values, z, run_name,
                                                            evaluation_subdirectory)
                print(f"Optimal x: {optimal_x}, Optimal y: {optimal_y}")

            print("Evaluation complete. Preparing to plot final results...")
            print(
                f"Data lengths: allowed={len(all_allowed_values)}, infected={len(all_infected_values)}, community_risk={len(all_community_risk_values)}")

            # Call the plotting function with all accumulated data

            self.plot_evaluation_results(
                all_allowed_values,
                all_infected_values,
                all_community_risk_values,
                run_name,
                evaluation_subdirectory
            )
            final_states = self.simulate_steady_state(alpha=alpha)
            self.plot_simulated_steady_state(final_states, run_name, alpha)
            # Plotting
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

            # Subplot 1: Community Risk and Allowed Values
            ax1.set_xlabel('Week')
            ax1.set_ylabel('Community Risk', color='tab:green')
            ax1.plot(range(1, len(self.community_risk_values) + 1), self.community_risk_values, marker='s',
                     linestyle='--',
                     color='tab:green', label='Community Risk')
            ax1.tick_params(axis='y', labelcolor='tab:green')

            ax1b = ax1.twinx()
            ax1b.set_ylabel('Allowed Values', color='tab:orange')
            ax1b.bar(range(1, len(allowed_values_over_time) + 1), allowed_values_over_time, color='tab:orange',
                     alpha=0.6,
                     width=0.4, align='center', label='Allowed')
            ax1b.tick_params(axis='y', labelcolor='tab:orange')

            ax1.legend(loc='upper left')
            ax1b.legend(loc='upper right')

            # Subplot 2: Infected Students Over Time
            ax2.set_xlabel('Week')
            ax2.set_ylabel('Number of Infected Students', color='tab:blue')
            ax2.plot(range(1, len(infected_values_over_time) + 1), infected_values_over_time, marker='o', linestyle='-',
                     color='tab:blue', label='Infected')
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            ax2.legend(loc='upper left')

            # Set x-ticks to show fewer labels and label weeks from 1 to n
            ticks = range(0, len(self.community_risk_values),
                          max(1, len(self.community_risk_values) // 10))  # Show approximately 10 ticks
            labels = [f'Week {i + 1}' for i in ticks]

            ax1.set_xticks(ticks)
            ax1.set_xticklabels(labels, rotation=45)
            ax2.set_xticks(ticks)
            ax2.set_xticklabels(labels, rotation=45)

            # Adjust layout and save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(evaluation_subdirectory, f"evaluation_plot_{run_name}.png"))
            plt.show()
            print("Final plotting complete.")
        return total_rewards

    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def compute_tolerance_interval(self, data, alpha, beta):
        """
        Compute the (alpha, beta)-tolerance interval for a given data sample.

        Parameters:
        data (list or numpy array): The data sample.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence level).
        beta (float): The proportion of future samples to be captured (e.g., 0.9 for 90% of the population).

        Returns:
        (float, float): The lower and upper bounds of the tolerance interval.
        """
        n = len(data)
        if n == 0:
            return np.nan, np.nan  # Handle case with no data

        sorted_data = np.sort(data)

        # Compute the number of samples that do not belong to the middle beta proportion
        nu = stats.binom.ppf(1 - alpha, n, beta)
        nu = int(nu)

        if nu >= n:
            return sorted_data[0], sorted_data[-1]  # If nu is greater than available data points, return full range

        # Compute the indices for the lower and upper bounds
        l = int(np.floor((n - nu) / 2))
        u = int(np.ceil(n - (n - nu) / 2))

        return sorted_data[l], sorted_data[u]

    def visualize_tolerance_interval_curve(self, returns_per_episode, alpha, beta, output_path, metric='mean'):
        """
        Visualize the (alpha, beta)-tolerance interval curve over episodes for mean or median performance.

        Parameters:
        returns_per_episode (list): The list of returns per episode across multiple runs.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence level).
        beta (float): The proportion of future samples to be captured (e.g., 0.9 for 90% of the population).
        output_path (str): The file path to save the plot.
        metric (str): The metric to visualize ('mean' or 'median').
        """
        num_episodes = len(returns_per_episode[0])
        lower_bounds = []
        upper_bounds = []
        central_tendency = []
        episodes = list(range(num_episodes))

        for episode in episodes:
            returns_at_episode = [returns[episode] for returns in
                                  returns_per_episode]  # Shape: (num_runs, episode_length)
            returns_at_episode = [item for sublist in returns_at_episode for item in sublist]  # Flatten to 1D

            if metric == 'mean':
                performance = np.mean(returns_at_episode)
            elif metric == 'median':
                performance = np.median(returns_at_episode)
            else:
                raise ValueError("Invalid metric specified. Use 'mean' or 'median'.")

            central_tendency.append(performance)
            lower, upper = self.compute_tolerance_interval(returns_at_episode, alpha, beta)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        central_tendency = np.array(central_tendency)

        # Apply moving average
        window_size = 100  # Set the window size for moving average
        central_tendency_smooth = self.moving_average(central_tendency, window_size)
        lower_bounds_smooth = self.moving_average(lower_bounds, window_size)
        upper_bounds_smooth = self.moving_average(upper_bounds, window_size)
        episodes_smooth = range(len(central_tendency_smooth))

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot central tendency
        sns.lineplot(x=episodes_smooth, y=central_tendency_smooth, color='blue',
                     label=f'{metric.capitalize()} Performance')

        # Fill between for tolerance interval
        plt.fill_between(episodes_smooth, lower_bounds_smooth, upper_bounds_smooth, color='lightblue', alpha=0.2,
                         label=f'Tolerance Interval (α={alpha}, β={beta})')

        plt.title(f'Tolerance Interval Curve for {metric.capitalize()} Performance')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.legend()
        plt.savefig(output_path)
        plt.close()

    def compute_confidence_interval(self, data, alpha):
        """
        Compute the confidence interval for a given data sample using the Student t-distribution.

        Parameters:
        data (list or numpy array): The data sample.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence interval).

        Returns:
        (float, float): The lower and upper bounds of the confidence interval.
        """
        n = len(data)
        if n == 0:
            return np.nan, np.nan  # Handle case with no data

        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin_of_error = t_value * std_err
        return mean - margin_of_error, mean + margin_of_error

    def visualize_confidence_interval(self, returns, alpha, output_path):
        """
        Visualize the confidence interval over episodes.

        Parameters:
        returns (list): The list of returns per episode across multiple runs.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence interval).
        output_path (str): The file path to save the plot.
        """
        window_size = 100  # Set the window size for moving average
        means = []
        lower_bounds = []
        upper_bounds = []
        episodes = list(range(len(returns[0])))  # Assume all runs have the same number of episodes

        for episode in episodes:
            episode_returns = [returns[run][episode] for run in range(len(returns))]
            mean = np.mean(episode_returns)
            lower, upper = self.compute_confidence_interval(episode_returns, alpha)
            means.append(mean)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        means = np.array(means)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        # Apply moving average
        means_smooth = self.moving_average(means, window_size)
        lower_bounds_smooth = self.moving_average(lower_bounds, window_size)
        upper_bounds_smooth = self.moving_average(upper_bounds, window_size)
        episodes_smooth = range(len(means_smooth))

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Plot mean performance
        sns.lineplot(x=episodes_smooth, y=means_smooth, label='Mean Performance', color='orange')

        # Fill between for confidence interval
        plt.fill_between(episodes_smooth, lower_bounds_smooth, upper_bounds_smooth, color='#FFD580', alpha=0.2,
                         label=f'Confidence Interval (α={alpha})')

        plt.title(f'Confidence Interval Curve for Mean Performance')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.legend()
        plt.savefig(output_path)
        plt.close()

    def visualize_boxplot_confidence_interval(self, returns, alpha, output_path):
        """
        Visualize the confidence interval using box plots.

        Parameters:
        returns (list): The list of returns per episode across multiple runs.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence interval).
        output_path (str): The file path to save the plot.
        """
        num_episodes = len(returns[0])
        num_runs = len(returns)

        # Create a DataFrame for easier plotting with seaborn
        data = []
        for run in range(num_runs):
            for episode in range(num_episodes):
                # Flatten the list of returns if it's a nested list
                if isinstance(returns[run][episode], (list, np.ndarray)):
                    for ret in returns[run][episode]:
                        data.append([episode, ret])
                else:
                    data.append([episode, returns[run][episode]])

        df = pd.DataFrame(data, columns=["Episode", "Return"])

        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")

        # Plot the boxplot
        sns.boxplot(x="Episode", y="Return", data=df, whis=[100 * alpha / 2, 100 * (1 - alpha / 2)], color='lightblue')
        plt.title(f'Box Plot of Returns with Confidence Interval (α={alpha})')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.xticks(ticks=range(num_episodes), labels=range(num_episodes))
        plt.savefig(output_path)
        plt.close()

    def train_single_run(self, alpha):
        """Train the agent."""
        rewards_per_episode = []
        reward_history = []

        for episode in tqdm(range(self.max_episodes)):
            self.decay_handler.set_decay_function(self.decay_function)
            state = self.env.reset()
            c_state = state[0]
            terminated = False
            e_return = []
            step = 0



            while not terminated:
                action = self._policy('train', c_state)
                converted_state = str(tuple(c_state))
                state_idx = self.all_states.index(converted_state)
                # Convert the action index back to an action list
                action_list = self.index_to_action_list(action, self.env.action_space.nvec)

                # Ensure action_list contains integers, not arrays
                action_list = [int(a[0]) if isinstance(a, np.ndarray) else int(a) for a in action_list]

                # Scale the actions using the scale_action function
                scaled_action = [self.scale_action(a, n) for a, n in zip(action_list, self.env.action_space.nvec)]

                action_alpha_list = [*scaled_action, alpha]

                # Execute the action and observe the next state and reward
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                next_state = self.discretize_state(next_state)
                action_idx = self.action_list_to_index(action_list, self.env.action_space.nvec)
                # print('action index:', action_idx)
                if action_idx >= self.q_table.shape[1]:
                    raise IndexError(
                        f"Action index {action_idx} is out of bounds for Q-table with shape {self.q_table.shape}")

                try:
                    old_value = self.q_table[state_idx, action_idx]
                except IndexError as e:
                    logging.error(f"IndexError in Q-table access: {e}")
                    raise

                next_max = np.max(self.q_table[self.all_states.index(str(tuple(next_state)))])
                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                        reward + self.discount_factor * next_max)
                self.q_table[state_idx, action_idx] = new_value

                step += 1
                c_state = next_state
                week_reward = int(reward)
                e_return.append(week_reward)
                reward_history.append(reward)

            avg_episode_return = sum(e_return) / len(e_return)
            rewards_per_episode.append(e_return)  # Append the list of rewards per episode

            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)

        print("Training complete.")
        return rewards_per_episode

    def multiple_runs(self, num_runs, alpha_t, beta_t):
        returns_per_episode = []

        for run in range(num_runs):
            self.q_table = np.zeros_like(self.q_table)  # Reset Q-table for each run
            returns = self.train_single_run(alpha_t)
            returns_per_episode.append(returns)

        # Ensure returns_per_episode is correctly structured
        returns_per_episode = np.array(returns_per_episode)  # Shape: (num_runs, max_episodes, episode_length)

        output_path_mean = os.path.join(self.results_subdirectory, 'tolerance_interval_mean.png')
        output_path_median = os.path.join(self.results_subdirectory, 'tolerance_interval_median.png')

        self.visualize_tolerance_interval_curve(returns_per_episode, alpha_t, beta_t, output_path_mean, 'mean')
        self.visualize_tolerance_interval_curve(returns_per_episode, alpha_t, beta_t, output_path_median, 'median')

        wandb.log({"Tolerance Interval Mean": [wandb.Image(output_path_mean)]})
        wandb.log({"Tolerance Interval Median": [wandb.Image(output_path_median)]})

        # Confidence Intervals
        confidence_alpha = 0.05  # 95% confidence interval
        confidence_output_path = os.path.join(self.results_subdirectory, 'confidence_interval.png')
        self.visualize_confidence_interval(returns_per_episode, confidence_alpha, confidence_output_path)
        wandb.log({"Confidence Interval": [wandb.Image(confidence_output_path)]})

        # Box Plot Confidence Intervals
        # boxplot_output_path = os.path.join(self.results_subdirectory, 'boxplot_confidence_interval.png')
        # self.visualize_boxplot_confidence_interval(returns_per_episode, confidence_alpha, boxplot_output_path)
        # wandb.log({"Box Plot Confidence Interval": [wandb.Image(boxplot_output_path)]})

        # Calculate and print the mean reward in the last episode across all runs
        last_episode_rewards = [returns[-1] for returns in returns_per_episode]
        mean_last_episode_reward = np.mean(last_episode_rewards)
        print(f"Mean reward in the last episode across all runs: {mean_last_episode_reward}")

    def eval_with_csv(self, alpha, episodes, csv_path):
        """Evaluate the trained agent using community risk values from a CSV file."""

        # Read the community risk values from the CSV file
        community_risk_df = pd.read_csv(csv_path)
        community_risk_values = community_risk_df['community_risk'].tolist()

        total_class_capacity_utilized = 0
        last_action = None
        policy_changes = 0
        total_reward = 0
        rewards = []
        infected_dict = {}
        allowed_dict = {}
        rewards_dict = {}
        community_risk_dict = {}
        eval_dir = 'evaluation'
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        eval_file_path = os.path.join(eval_dir, f'eval_policies_data_aaai_multi.csv')
        # Check if the file exists already. If not, create it and add the header
        if not os.path.isfile(eval_file_path):
            with open(eval_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header to the CSV file
                writer.writerow(['Alpha', 'Episode', 'Step', 'Infections', 'Allowed', 'Reward', 'CommunityRisk'])

        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            c_state = state[0]
            terminated = False
            episode_reward = 0
            episode_infections = 0
            infected = []
            allowed = []
            community_risk = []
            eps_rewards = []

            for step, comm_risk in enumerate(community_risk_values):
                if terminated:
                    break
                converted_state = str(tuple(c_state))
                state_idx = self.all_states.index(converted_state)

                # Select an action based on the Q-table
                action = np.argmax(self.q_table[state_idx])

                list_action = list(eval(self.all_actions[action]))
                c_list_action = [i * 50 for i in list_action]

                action_alpha_list = [*c_list_action, alpha]
                self.env.campus_state.community_risk = comm_risk  # Set community risk value

                # Execute the action and observe the next state and reward
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                eps_rewards.append(reward)
                infected.append(info['infected'])
                allowed.append(info['allowed'])
                community_risk.append(info['community_risk'])
                episode_infections += sum(info['infected'])

                # Update policy stability metrics
                if last_action is not None and last_action != action:
                    policy_changes += 1
                last_action = action

                # Update class utilization metrics
                total_class_capacity_utilized += sum(info['allowed'])

                # Update the state to the next state
                c_state = next_state

            infected_dict[episode] = infected
            allowed_dict[episode] = allowed
            rewards_dict[episode] = eps_rewards
            community_risk_dict[episode] = community_risk

        print("infected: ", infected_dict, "allowed: ", allowed_dict, "rewards: ", rewards_dict, "community_risk: ", community_risk_dict)
        for episode in infected_dict:
            plt.figure(figsize=(15, 5))

            # Flatten the list of lists for infections and allowed students
            infections = [inf[0] for inf in infected_dict[episode]] if episode in infected_dict else []
            allowed_students = [alw[0] for alw in allowed_dict[episode]] if episode in allowed_dict else []
            rewards = rewards_dict[episode] if episode in rewards_dict else []
            community_risk = community_risk_dict[episode] if episode in community_risk_dict else []

            # Convert range to numpy array for element-wise operations
            steps = np.arange(len(infections))

            # Define bar width and offset
            bar_width = 0.4
            offset = bar_width / 4

            # Bar plot for infections
            plt.bar(steps - offset, infections, width=bar_width, label='Infections', color='#bc5090', align='center')

            # Bar plot for allowed students
            plt.bar(steps + offset, allowed_students, width=bar_width, label='Allowed Students', color='#003f5c',
                    alpha=0.5, align='edge')

            # Line plot for rewards
            plt.plot(steps, rewards, label='Rewards', color='#ffa600', linestyle='-', marker='o')

            plt.xlabel('Step')
            plt.ylabel('Count')
            plt.title(f'Evaluation of agent {self.run_name} Policy for {episode} episodes')
            plt.legend()

            plt.tight_layout()

            # Save the figure
            fig_path = os.path.join(eval_dir, f'{self.run_name}_metrics.png')
            plt.savefig(fig_path)
            print(f"Figure saved to {fig_path}")

            plt.close()  # Close the figure to free up memory

        with open(eval_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Iterate over each episode and step to append the data
            for episode in tqdm(range(episodes)):
                for step in range(len(infected_dict[episode])):
                    writer.writerow([
                        alpha,
                        episode,
                        step,
                        infected_dict[episode][step],
                        allowed_dict[episode][step],
                        rewards_dict[episode][step],
                        community_risk_dict[episode][step]
                    ])

        print(f"Data for alpha {alpha} appended to {eval_file_path}")

        return infected_dict, allowed_dict, rewards_dict, community_risk_dict
