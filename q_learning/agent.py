import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import itertools
from .utilities import load_config
from .visualizer import visualize_all_states, visualize_q_table, visualize_variance_in_rewards_heatmap, \
    visualize_explained_variance, visualize_variance_in_rewards, visualize_infected_vs_community_risk_table, states_visited_viz
import os
import io
import json
import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import random
import pandas as pd
import csv
import math
import collections
import seaborn as sns
from scipy import stats
from scipy.interpolate import make_interp_spline
import time

SEED = 100
random.seed(SEED)
np.random.seed(SEED)
epsilon = 1e-10

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
            exploration_rate = self.initial_exploration_rate * np.exp(-episode / self.max_episodes)

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

        print("Q-table initialized from CSV.")

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
                q_values = self.q_table[state_idx]
                action = np.argmax(q_values)
            else:
                action = random.randint(0, self.q_table.shape[1] - 1)
        elif mode == 'test':
            action = np.argmax(self.q_table[state_idx])

        # Convert single action index to list of actions for each course
        num_courses = len(self.env.action_space.nvec)
        course_actions = [action % 3 for _ in range(num_courses)]  # Assuming 3 actions per course
        return course_actions

    def train(self, alpha):
        """Train the agent."""
        start_time = time.time()
        actual_rewards = []
        predicted_rewards = []
        rewards_per_episode = []
        last_episode = {}
        visited_state_counts = {}
        q_value_history = []
        reward_history = []
        td_errors = []
        training_log = []
        cumulative_rewards = []
        q_value_diffs = []

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

            while not terminated:
                action = self._policy('train', c_state)
                converted_state = str(tuple(c_state))
                state_idx = self.all_states.index(converted_state)

                episode_visited_states.add(converted_state)

                c_list_action = [i * 50 for i in action]
                action_alpha_list = [*c_list_action, alpha]

                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                action_idx = sum([a * (3 ** i) for i, a in enumerate(action)])
                old_value = self.q_table[state_idx, action_idx]
                next_max = np.max(self.q_table[self.all_states.index(str(tuple(next_state)))])
                new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
                        reward + self.discount_factor * next_max)
                self.q_table[state_idx, action_idx] = new_value

                td_error = abs(reward + self.discount_factor * next_max - old_value)
                episode_td_errors.append(td_error)

                if last_action is not None and last_action != action:
                    policy_changes += 1
                last_action = action

                predicted_reward = self.q_table[state_idx, action]

                q_values_list.append(self.q_table[state_idx])

                self.state_action_visits[state_idx, action] += 1
                self.state_visits[state_idx] += 1

                total_reward += reward
                e_return.append(reward)

                step += 1
                c_state = next_state

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
            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)

        csvfile.close()
        print("Training complete.")
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

    def evaluate(self, run_name, num_episodes=1, alpha=0.5, csv_path=None):
        policy_dir = self.shared_config['directories']['policy_directory']
        q_table_path = os.path.join(policy_dir, f'q_table_{run_name}.npy')
        results_directory = self.results_subdirectory

        if not os.path.exists(q_table_path):
            raise FileNotFoundError(f"Q-table file not found in {q_table_path}")

        self.q_table = np.load(q_table_path)
        print(f"Loaded Q-table from {q_table_path}")

        total_rewards = []
        allowed_values_over_time = []
        infected_values_over_time = []

        # Prepare the CSV file for writing
        evaluation_subdirectory = os.path.join(results_directory, run_name)
        os.makedirs(evaluation_subdirectory, exist_ok=True)
        csv_file_path = os.path.join(evaluation_subdirectory, f'evaluation_metrics_{run_name}.csv')

        if csv_path:
            self.community_risk_values = self.read_community_risk_from_csv(csv_path)
            self.max_weeks = len(self.community_risk_values)
            print(f"Community Risk Values: {self.community_risk_values}")

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Step', 'State', 'Action', 'Total Reward'])

            for episode in range(num_episodes):
                state, _ = self.env.reset()
                print(f"Initial state for episode {episode + 1}: {state}")
                c_state = state
                terminated = False
                total_reward = 0
                step = 0

                # Initialize plotting lists with the initial state
                allowed_values_over_time.append(0)  # No action taken at the initial state
                infected_values_over_time.append(c_state[0] * 10)  # Scale from 0-10 to 0-100

                while not terminated:
                    action = self._policy('test', c_state)
                    print('State:', c_state)
                    c_list_action = [i * 50 for i in action]  # scale 0, 1, 2 to 0, 50, 100
                    action_alpha_list = [*c_list_action, alpha]
                    next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                    c_state = next_state
                    total_reward += reward

                    # Write the step's total reward to the CSV file
                    writer.writerow([episode + 1, step + 1, c_state[0] * 10, c_list_action[0], total_reward])
                    step += 1

                    # Collect data for plotting
                    if episode == 0:  # Only collect data for the first episode for simplicity
                        allowed_values_over_time.append(c_list_action[0])
                        infected_values_over_time.append(c_state[0] * 10)  # Scale from 0-10 to 0-100

                total_rewards.append(total_reward)
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        # Subplot 1: Community Risk and Allowed Values
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Community Risk', color='tab:green')
        ax1.plot(range(1, len(self.community_risk_values) + 1), self.community_risk_values, marker='s', linestyle='--',
                 color='tab:green', label='Community Risk')
        ax1.tick_params(axis='y', labelcolor='tab:green')

        ax1b = ax1.twinx()
        ax1b.set_ylabel('Allowed Values', color='tab:orange')
        ax1b.bar(range(1, len(allowed_values_over_time) + 1), allowed_values_over_time, color='tab:orange', alpha=0.6,
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

        return avg_reward

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
        sns.lineplot(x=episodes_smooth, y=means_smooth, label='Mean Performance', color='blue')

        # Fill between for confidence interval
        plt.fill_between(episodes_smooth, lower_bounds_smooth, upper_bounds_smooth, color='lightblue', alpha=0.2,
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
                c_list_action = [i * 50 for i in action]  # scale 0, 1, 2 to 0, 50, 100

                action_alpha_list = [*c_list_action, alpha]

                # Execute the action and observe the next state and reward
                next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
                action_idx = sum([a * (3 ** i) for i, a in enumerate(action)])  # Convert action list to single index
                old_value = self.q_table[state_idx, action_idx]
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
