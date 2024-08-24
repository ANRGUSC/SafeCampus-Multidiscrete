import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import logging
from collections import deque
import random
import itertools
from tqdm import tqdm
from .utilities import load_config
from .visualizer import visualize_all_states, states_visited_viz
import pandas as pd

import wandb
from torch.optim.lr_scheduler import StepLR
import math
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import torch.nn.functional as F
import csv
import os
import time
epsilon = 1e-10

def log_metrics_to_csv(file_path, metrics):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seed for reproducibility
set_seed(100)  # Replace 42 with your desired seed value


def log_all_states_visualizations(q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory):
    file_paths = visualize_all_states(q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory)

    # Log all generated visualizations
    # wandb_images = [wandb.Image(path) for path in file_paths]
    # wandb.log({"All States Visualization": wandb_images})
    #
    # # Log them individually with dimension information
    # for path in file_paths:
    #     infected_dim = path.split('infected_dim_')[-1].split('.')[0]
    #     wandb.log({f"All States Visualization (Infected Dim {infected_dim})": wandb.Image(path)})


def log_states_visited(states, visit_counts, alpha, results_subdirectory):
    file_paths = states_visited_viz(states, visit_counts, alpha, results_subdirectory)

    # Log all generated heatmaps
    # wandb_images = [wandb.Image(path) for path in file_paths]
    # wandb.log({"States Visited": wandb_images})
    #
    # # Log them individually with dimension information
    # for path in file_paths:
    #     if "error" in path:
    #         wandb.log({"States Visited Error": wandb.Image(path)})
    #     else:
    #         dim = path.split('infected_dim_')[-1].split('.')[0]
    #         wandb.log({f"States Visited (Infected Dim {dim})": wandb.Image(path)})

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
                np.pi * episode / self.max_episodes) ** 2
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


# Define the Lyapunov Network
class LyapunovNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LyapunovNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(DeepQNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h_prime = self.encoder(x)
        Q_values = self.out(h_prime)
        return Q_values
class DQNCustomAgent:
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
        self.results_directory = self.shared_config.get('directories', {}).get('results_directory', None)

        # Debugging: Print results_directory and run_name
        print(f"results_directory: {self.results_directory}")
        print(f"run_name: {run_name}")

        # Create a unique subdirectory for each run to avoid overwriting results
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.agent_type = "dqn_custom"
        self.run_name = run_name
        self.results_subdirectory = os.path.join(self.results_directory, self.agent_type, self.run_name)
        if not os.path.exists(self.results_subdirectory):
            os.makedirs(self.results_subdirectory, exist_ok=True)
        self.model_directory = self.shared_config['directories']['model_directory']

        self.model_subdirectory = os.path.join(self.model_directory, self.agent_type, self.run_name)
        if not os.path.exists(self.model_subdirectory):
            os.makedirs(self.model_subdirectory, exist_ok=True)

        # Set up logging to the correct directory
        log_file_path = os.path.join(self.results_subdirectory, 'agent_log.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO)

        # Initialize wandb
        wandb.init(project=self.agent_type, name=self.run_name)
        self.env = env

        # Initialize the neural network
        self.input_dim = len(env.reset()[0])
        self.output_dim = env.action_space.nvec[0]
        self.hidden_dim = self.agent_config['agent']['hidden_units']
        self.num_courses = self.env.action_space.nvec[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepQNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])

        # Initialize agent-specific configurations and variables

        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']
        self.target_network_frequency = self.agent_config['agent']['target_network_frequency']

        # Replay memory
        # self.replay_memory = deque(maxlen=self.agent_config['agent']['replay_memory_capacity'])
        # self.batch_size = self.agent_config['agent']['batch_size']

        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]

        # moving average for early stopping criteria
        self.moving_average_window = 100  # Number of episodes to consider for moving average
        self.stopping_criterion = 0.01  # Threshold for stopping
        self.prev_moving_avg = -float(
            'inf')  # Initialize to negative infinity to ensure any reward is considered an improvement in the first episode.

        # Hidden State
        self.hidden_state = None
        self.reward_window = deque(maxlen=self.moving_average_window)
        # Initialize the learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=200, gamma=0.9)
        self.learning_rate_decay = self.agent_config['agent']['learning_rate_decay']

        self.softmax_temperature = self.agent_config['agent']['softmax_temperature']

        self.state_visit_counts = {}
        self.noise_scale = 0.1
        self.noise_decay_rate = 0.9999
        self.noise_frequency = 1000
        self.aggressiveness = 0.8

        self.decay_handler = ExplorationRateDecay(self.max_episodes, self.min_exploration_rate, self.exploration_rate)
        self.decay_function = self.agent_config['agent']['e_decay_function']

        csv_file_name = f'training_metrics_dqn_{self.run_name}.csv'
        # CSV file for metrics
        self.csv_file_path = os.path.join(self.results_subdirectory, csv_file_name)
        self.time_complexity = 0
        self.convergence_rate = 0
        self.policy_entropy = 0

        # Handle CSV input
        if csv_path:
            self.community_risk_values = self.read_community_risk_from_csv(csv_path)
            self.max_weeks = len(self.community_risk_values)
            print(f"Community Risk Values: {self.community_risk_values}")
        else:
            self.community_risk_values = None
            self.max_weeks = self.env.campus_state.model.get_max_weeks()

    def read_community_risk_from_csv(self, csv_path):
        try:
            community_risk_df = pd.read_csv(csv_path)
            return community_risk_df['Risk-Level'].tolist()
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

    def select_action(self, state):
        if random.random() < self.exploration_rate:
            return [random.randint(0, self.output_dim - 1) * 50 for _ in range(self.num_courses)]
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state)
                # print(f"Q-values shape in select_action: {q_values.shape}")
                # Repeat Q-values for each course
                q_values = q_values.repeat(1, self.num_courses).view(self.num_courses, -1)

                actions = q_values.max(1)[1].tolist()
                return [action * 50 for action in actions]

    def calculate_convergence_rate(self, episode_rewards):
        # Simple example: Convergence rate is the change in reward over the last few episodes
        if len(episode_rewards) < 10:
            return 0
        return np.mean(np.diff(episode_rewards[-10:]))  # Change in reward over the last 10 episodes

    def calculate_policy_entropy(self, q_values):
        # Convert to PyTorch tensor if it's a NumPy array
        if isinstance(q_values, np.ndarray):
            q_values = torch.from_numpy(q_values).float()

        # Ensure q_values is on the correct device
        q_values = q_values.to(self.device)

        # Ensure q_values is 2D
        if q_values.dim() == 1:
            q_values = q_values.unsqueeze(0)

        # Apply softmax to get probabilities
        policy = F.softmax(q_values, dim=-1)

        # Calculate entropy
        log_policy = torch.log(policy + 1e-10)  # Add small constant to avoid log(0)
        entropy = -torch.sum(policy * log_policy, dim=-1)

        # Return mean entropy as a Python float
        return entropy.mean().item()


    def train(self, alpha):
        start_time = time.time()
        pbar = tqdm(total=self.max_episodes, desc="Training Progress", leave=True)

        actual_rewards = []
        predicted_rewards = []
        visited_state_counts = {}
        explained_variance_per_episode = []

        # Initialize accumulators for allowed and infected
        allowed_means_per_episode = []
        infected_means_per_episode = []

        file_exists = os.path.isfile(self.csv_file_path)
        csvfile = open(self.csv_file_path, 'a', newline='')
        writer = csv.DictWriter(csvfile,
                                fieldnames=['episode', 'cumulative_reward', 'average_reward', 'discounted_reward',
                                            'convergence_rate', 'sample_efficiency', 'policy_entropy',
                                            'space_complexity'])
        if not file_exists:
            writer.writeheader()

        previous_q_values = None

        for episode in range(self.max_episodes):
            self.decay_handler.set_decay_function(self.decay_function)
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            done = False
            episode_rewards = []
            visited_states = set()  # Using a set to track unique states
            episode_q_values = []
            step = 0
            q_value_change = 0
            episode_allowed = []
            episode_infected = []

            while not done:
                actions = self.select_action(state)
                next_state, reward, done, _, info = self.env.step((actions, alpha))
                next_state = np.array(next_state, dtype=np.float32)

                # Update the total allowed and infected counts
                episode_allowed.append(sum(info.get('allowed', [])))  # Sum all courses' allowed students
                episode_infected.append(sum(info.get('infected', [])))

                original_actions = [action // 50 for action in actions]
                total_reward += reward
                episode_rewards.append(reward)

                current_q_values = self.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
                if previous_q_values is not None:
                    q_value_change += np.mean((current_q_values - previous_q_values) ** 2)
                previous_q_values = current_q_values
                # Update Q-values directly without replay memory or batch processing
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                action_tensor = torch.LongTensor(original_actions)

                current_q_values = self.model(state_tensor)
                next_q_values = self.model(next_state_tensor)
                # print(info)

                # Handle multi-course scenario
                num_courses = len(original_actions)
                current_q_values = current_q_values.repeat(1, num_courses).view(num_courses, -1)
                next_q_values = next_q_values.repeat(1, num_courses).view(num_courses, -1)

                current_q_values = current_q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                next_q_values = next_q_values.max(1)[0]

                target_q_values = reward + (1 - done) * self.discount_factor * next_q_values

                loss = nn.MSELoss()(current_q_values, target_q_values)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                episode_q_values.extend(current_q_values.detach().numpy().tolist())

                state = next_state
                state_tuple = tuple(state)
                visited_states.add(state_tuple)
                visited_state_counts[state_tuple] = visited_state_counts.get(state_tuple, 0) + 1
                step += 1

            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)
            e_mean_allowed = sum(episode_allowed) / len(episode_allowed)
            e_mean_infected = sum(episode_infected) / len(episode_infected)
            allowed_means_per_episode.append(e_mean_allowed)
            infected_means_per_episode.append(e_mean_infected)
            actual_rewards.append(episode_rewards)
            predicted_rewards.append(episode_q_values)
            avg_episode_return = sum(episode_rewards) / len(episode_rewards)
            cumulative_reward = sum(episode_rewards)
            discounted_reward = sum([r * (self.discount_factor ** i) for i, r in enumerate(episode_rewards)])
            sample_efficiency = len(visited_states)  # Unique states visited in the episode
            policy_entropy = self.calculate_policy_entropy(
                self.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy())

            metrics = {
                'episode': episode,
                'cumulative_reward': cumulative_reward,
                'average_reward': avg_episode_return,
                'discounted_reward': discounted_reward,
                'sample_efficiency': sample_efficiency,
                'policy_entropy': policy_entropy,
                'space_complexity': sum(p.numel() for p in self.model.parameters())
            }
            writer.writerow(metrics)
            # wandb.log({'cumulative_reward': cumulative_reward})

            pbar.update(1)
            pbar.set_description(f"Total Reward: {total_reward:.2f}, Epsilon: {self.exploration_rate:.2f}")

        pbar.close()
        csvfile.close()

        # Calculate the means for allowed and infected
        mean_allowed = round(sum(allowed_means_per_episode) / len(allowed_means_per_episode))
        mean_infected = round(sum(infected_means_per_episode) / len(infected_means_per_episode))

        # Save the results in a separate CSV file
        summary_file_path = os.path.join(self.results_subdirectory, 'mean_allowed_infected.csv')
        with open(summary_file_path, 'w', newline='') as summary_csvfile:
            summary_writer = csv.DictWriter(summary_csvfile, fieldnames=['mean_allowed', 'mean_infected'])
            summary_writer.writeheader()
            summary_writer.writerow({'mean_allowed': mean_allowed, 'mean_infected': mean_infected})


        model_file_path = os.path.join(self.model_subdirectory, f'model.pt')
        torch.save(self.model.state_dict(), model_file_path)
        print(f"Model saved at: {model_file_path}")

        self.log_states_visited(list(visited_state_counts.keys()), list(visited_state_counts.values()), alpha,
                                self.results_subdirectory)
        self.log_all_states_visualizations(self.model, self.run_name, self.max_episodes, alpha,
                                           self.results_subdirectory)

        return self.model

    def generate_all_states(self):
        value_range = range(0, 101, 10)
        input_dim = self.model.encoder[0].in_features

        if input_dim == 2:
            # If the model expects only 2 inputs, we'll use the first course and community risk
            all_states = [np.array([i, j]) for i in value_range for j in value_range]
        else:
            # Generate states for all courses and community risk
            course_combinations = itertools.product(value_range, repeat=self.num_courses)
            all_states = [np.array(list(combo) + [risk]) for combo in course_combinations for risk in value_range]

            # Truncate or pad states to match input_dim
            all_states = [state[:input_dim] if len(state) > input_dim else
                          np.pad(state, (0, max(0, input_dim - len(state))), 'constant')
                          for state in all_states]

        return all_states

    def log_all_states_visualizations(self, model, run_name, max_episodes, alpha, results_subdirectory):
        all_states = self.generate_all_states()
        num_courses = len(self.env.students_per_course)
        file_paths = visualize_all_states(model, all_states, run_name, num_courses, max_episodes, alpha,
                                          results_subdirectory, self.env.students_per_course)
        print("file_paths: ", file_paths)

    def log_states_visited(self, states, visit_counts, alpha, results_subdirectory):
        file_paths = states_visited_viz(states, visit_counts, alpha, results_subdirectory)
        print("file_paths: ", file_paths)

    def calculate_explained_variance(self, y_true, y_pred):
        """
        Calculate the explained variance.

        :param y_true: array-like of shape (n_samples,), Ground truth (correct) target values.
        :param y_pred: array-like of shape (n_samples,), Estimated target values.
        :return: float, Explained variance score.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) != len(y_pred):
            min_length = min(len(y_true), len(y_pred))
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]

        var_y = np.var(y_true)
        return np.mean(1 - np.var(y_true - y_pred) / var_y) if var_y != 0 else 0.0

    def train_single_run(self, seed, alpha):
        set_seed(seed)
        # Reset relevant variables for each run
        self.replay_memory = deque(maxlen=self.agent_config['agent']['replay_memory_capacity'])
        self.reward_window = deque(maxlen=self.moving_average_window)
        self.model = DeepQNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.target_model = DeepQNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=self.learning_rate_decay)

        self.run_rewards_per_episode = []  # Store rewards per episode for this run

        pbar = tqdm(total=self.max_episodes, desc=f"Training Run {seed}", leave=True)
        visited_state_counts = {}
        previous_q_values = None

        for episode in range(self.max_episodes):
            self.decay_handler.set_decay_function(self.decay_function)
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            done = False
            episode_rewards = []
            visited_states = set()  # Using a set to track unique states
            episode_q_values = []
            step = 0
            q_value_change = 0

            while not done:
                actions = self.select_action(state)
                next_state, reward, done, _, info = self.env.step((actions, alpha))
                next_state = np.array(next_state, dtype=np.float32)

                original_actions = [action // 50 for action in actions]
                total_reward += reward
                episode_rewards.append(reward)

                current_q_values = self.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
                if previous_q_values is not None:
                    q_value_change += np.mean((current_q_values - previous_q_values) ** 2)
                previous_q_values = current_q_values


                # Update Q-values directly without replay memory or batch processing
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                action_tensor = torch.LongTensor(original_actions)

                current_q_values = self.model(state_tensor)
                next_q_values = self.model(next_state_tensor)
                # print(info)

                # Handle multi-course scenario
                num_courses = len(original_actions)
                current_q_values = current_q_values.repeat(1, num_courses).view(num_courses, -1)
                next_q_values = next_q_values.repeat(1, num_courses).view(num_courses, -1)

                current_q_values = current_q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                next_q_values = next_q_values.max(1)[0]

                target_q_values = reward + (1 - done) * self.discount_factor * next_q_values

                loss = nn.MSELoss()(current_q_values, target_q_values)
                # print(info)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                episode_q_values.extend(current_q_values.detach().numpy().tolist())

                state = next_state
                state_tuple = tuple(state)
                visited_states.add(state_tuple)
                visited_state_counts[state_tuple] = visited_state_counts.get(state_tuple, 0) + 1
                step += 1

            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)
            self.run_rewards_per_episode.append(episode_rewards)


            pbar.update(1)
            pbar.set_description(f"Total Reward: {total_reward:.2f}, Epsilon: {self.exploration_rate:.2f}")

        pbar.close()


        # Debugging: Print all rewards after training
        print("All Rewards:", self.run_rewards_per_episode)

        return self.run_rewards_per_episode

    def moving_average(self, data, window_size):
        if len(data) < window_size:
            return data  # Not enough data to compute moving average, return original data
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
        l = int(np.floor(nu / 2))
        u = int(np.ceil(n - nu / 2))

        return sorted_data[l], sorted_data[u]

    def visualize_tolerance_interval_curve(self, returns_per_episode, alpha, beta, output_path, metric='mean'):
        num_episodes = len(returns_per_episode[0])
        lower_bounds = []
        upper_bounds = []
        central_tendency = []
        episodes = list(range(num_episodes))  # Assume all runs have the same number of episodes

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

        # Check data before smoothing
        print("Central Tendency (pre-smoothing):", central_tendency)
        print("Lower Bounds (pre-smoothing):", lower_bounds)
        print("Upper Bounds (pre-smoothing):", upper_bounds)

        # Smoothing the curve
        central_tendency_smooth = self.moving_average(central_tendency, 100)
        lower_bounds_smooth = self.moving_average(lower_bounds, 100)
        upper_bounds_smooth = self.moving_average(upper_bounds, 100)
        episodes_smooth = list(range(len(central_tendency_smooth)))

        # Check data after smoothing
        print("Central Tendency (post-smoothing):", central_tendency_smooth)
        print("Lower Bounds (post-smoothing):", lower_bounds_smooth)
        print("Upper Bounds (post-smoothing):", upper_bounds_smooth)

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

    def visualize_confidence_interval(self, returns, alpha, output_path):
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

        # Check data before smoothing
        print("Means (pre-smoothing):", means)
        print("Lower Bounds (pre-smoothing):", lower_bounds)
        print("Upper Bounds (pre-smoothing):", upper_bounds)

        # Smoothing the curve
        means_smooth = self.moving_average(means, 100)
        lower_bounds_smooth = self.moving_average(lower_bounds, 100)
        upper_bounds_smooth = self.moving_average(upper_bounds, 100)
        episodes_smooth = list(range(len(means_smooth)))

        # Check data after smoothing
        print("Means (post-smoothing):", means_smooth)
        print("Lower Bounds (post-smoothing):", lower_bounds_smooth)
        print("Upper Bounds (post-smoothing):", upper_bounds_smooth)

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
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)
        t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin_of_error = t_value * std_err
        return mean - margin_of_error, mean + margin_of_error


    def visualize_boxplot_confidence_interval(self, returns, alpha, output_path):
        """
        Visualize the confidence interval using box plots.

        Parameters:
        returns (list): The list of returns per episode across multiple runs.
        alpha (float): The nominal error rate (e.g., 0.05 for 95% confidence interval).
        output_path (str): The file path to save the plot.
        """
        episodes = list(range(len(returns[0])))  # Assume all runs have the same number of episodes
        returns_transposed = np.array(returns).T.tolist()  # Transpose to get returns per episode

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=returns_transposed, whis=[100 * alpha / 2, 100 * (1 - alpha / 2)], color='lightblue')
        plt.title(f'Box Plot of Returns with Confidence Interval (α={alpha})')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.xticks(ticks=range(len(episodes)), labels=episodes)
        plt.savefig(output_path)
        plt.close()

    def multiple_runs(self, num_runs, alpha_t, beta_t):
        returns_per_episode = []

        for run in range(num_runs):
            seed = int(run)
            returns = self.train_single_run(seed, alpha_t)
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






    # def evaluate(self, run_name, num_episodes=1, alpha=0.5, csv_path=None):
    #     model_subdirectory = os.path.join(self.model_directory, self.agent_type, run_name)
    #     model_file_path = os.path.join(model_subdirectory, f'model.pt')
    #     results_directory = self.results_subdirectory
    #
    #     if not os.path.exists(model_file_path):
    #         raise FileNotFoundError(f"Model file not found in {model_file_path}")
    #
    #     self.model.load_state_dict(torch.load(model_file_path))
    #     self.model.eval()
    #     print(f"Loaded model from {model_file_path}")
    #
    #     if csv_path:
    #         self.community_risk_values = self.read_community_risk_from_csv(csv_path)
    #         self.max_weeks = len(self.community_risk_values)
    #         print(f"Community Risk Values: {self.community_risk_values}")
    #
    #     total_rewards = []
    #     # allowed_values_over_time = []
    #     # infected_values_over_time = []
    #     evaluation_subdirectory = os.path.join(results_directory, run_name)
    #     os.makedirs(evaluation_subdirectory, exist_ok=True)
    #     csv_file_path = os.path.join(evaluation_subdirectory, f'evaluation_metrics_{run_name}.csv')
    #
    #     with open(csv_file_path, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(['Episode', 'Step', 'State', 'Action', 'Community Risk', 'Total Reward'])
    #
    #         for episode in range(num_episodes):
    #             # Initialize the state
    #             initial_infection_count = 20
    #             initial_community_risk = self.community_risk_values[0] if csv_path else 0
    #             state = np.array([initial_infection_count, int(initial_community_risk * 100)], dtype=np.float32)
    #             print(f"Initial state for episode {episode + 1}: {state}")
    #             total_reward = 0
    #             done = False
    #             step = 0
    #
    #             # Initialize plotting lists
    #             allowed_values_over_time = []
    #             infected_values_over_time = [initial_infection_count]
    #
    #             while not done:
    #                 with torch.no_grad():
    #                     state_tensor = torch.FloatTensor(state).unsqueeze(0)
    #                     q_values = self.model(state_tensor)
    #                     actions = q_values.argmax(dim=1).cpu().numpy()
    #                     actions = [action * 50 for action in actions]
    #
    #                 next_state, reward, done, _, info = self.env.step((actions, alpha))
    #                 next_state = np.array(next_state, dtype=np.float32)
    #                 total_reward += reward
    #
    #                 # Log the data
    #                 writer.writerow(
    #                     [episode + 1, step + 1, int(state[0]), actions[0], info['community_risk'], reward]
    #                 )
    #
    #                 # Collect data for plotting (now including the first week)
    #                 if episode == 0:
    #                     allowed_values_over_time.append(actions[0])
    #                     infected_values_over_time.append(next_state[0])  # Use next_state for infected count
    #
    #                 state = next_state
    #                 step += 1
    #
    #             total_rewards.append(total_reward)
    #
    #             print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    #
    #         avg_reward = np.mean(total_rewards)
    #     print(f"Cumulative Reward over {num_episodes} episodes: {sum(total_rewards)}")
    #
    #     # Debug prints
    #     print(f"Length of allowed_values_over_time: {len(allowed_values_over_time)}")
    #     print(f"Length of infected_values_over_time: {len(infected_values_over_time)}")
    #     print(f"Length of self.community_risk_values: {len(self.community_risk_values)}")
    #
    #     # Ensure all arrays have the same length
    #     min_length = min(len(allowed_values_over_time), len(infected_values_over_time), len(self.community_risk_values))
    #     allowed_values_over_time = allowed_values_over_time[:min_length]
    #     infected_values_over_time = infected_values_over_time[:min_length]
    #     community_risk_values = self.community_risk_values[:min_length]
    #
    #     # Create weeks array
    #     weeks = range(1, min_length + 1)
    #
    #     # Debug prints after adjustment
    #     print(f"After adjustment:")
    #     print(f"Length of allowed_values_over_time: {len(allowed_values_over_time)}")
    #     print(f"Length of infected_values_over_time: {len(infected_values_over_time)}")
    #     print(f"Length of community_risk_values: {len(community_risk_values)}")
    #     print(f"Length of weeks: {len(weeks)}")
    #
    #     # Plotting
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    #
    #     # Subplot 1: Community Risk
    #     sns.lineplot(x=weeks, y=community_risk_values,
    #                  marker='s', linestyle='--', color='darkgreen', linewidth=2.5, ax=ax1)
    #     ax1.set_xlabel('Week')
    #     ax1.set_ylabel('Community Risk', color='darkgreen')
    #     ax1.tick_params(axis='y', labelcolor='darkgreen')
    #
    #     # Subplot 2: Allowed and Infected Values
    #     ax2.bar(weeks, allowed_values_over_time,
    #             color='blue', alpha=0.6, width=0.4, align='center', label='Allowed')
    #     ax2.bar(weeks, infected_values_over_time,
    #             color='red', alpha=0.6, width=0.4, align='edge', label='Infected')
    #     ax2.set_xlabel('Week')
    #     ax2.set_ylabel('Allowed and Infected Values')
    #     ax2.legend(loc='upper left')
    #
    #     # Set x-ticks to show fewer labels and label weeks from 1 to n
    #     ticks = range(0, min_length, max(1, min_length // 10))
    #     labels = [f'Week {i + 1}' for i in ticks]
    #
    #     ax1.set_xticks(ticks)
    #     ax1.set_xticklabels(labels, rotation=45)
    #     ax2.set_xticks(ticks)
    #     ax2.set_xticklabels(labels, rotation=45)
    #
    #     # Adjust layout and save the plot
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(evaluation_subdirectory, f"evaluation_plot_{run_name}.png"))
    #     plt.show()
    #
    #     return avg_reward

    def identify_safety_set(self, allowed_values_over_time, infected_values_over_time, x, y, z,
                            evaluation_subdirectory):
        """Identify and plot the safety set based on constraints."""
        # Determine safety boundaries
        safe_infection_values = [val for val in infected_values_over_time if val <= x]
        safe_attendance_values = [val for val in allowed_values_over_time if val <= y]

        # Calculate the percentage of time spent in safe regions
        infection_safety_percentage = (len(safe_infection_values) / len(infected_values_over_time)) * 100
        attendance_safety_percentage = (len(safe_attendance_values) / len(allowed_values_over_time)) * 100

        # Save the safety percentages
        with open(os.path.join(evaluation_subdirectory, 'safety_set.txt'), 'w') as f:
            f.write(f"Safety Percentage for Infections <= {x}: {infection_safety_percentage}%\n")
            f.write(f"Safety Percentage for Attendance <= {y}: {attendance_safety_percentage}%\n")

        # Plot the safety set
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
        plt.show()

        return infection_safety_percentage >= (100 - z), attendance_safety_percentage >= (100 - z)

    def construct_cbf(self, allowed_values_over_time, infected_values_over_time, evaluation_subdirectory, x, y):
        """Construct and save the Control Barrier Function (CBF) based on the safety set."""
        # Define the CBFs
        B1 = lambda s: x - s[1]  # Safety for infections
        B2 = lambda s: y - s[0]  # Safety for attendance

        # Save the CBF equations
        with open(os.path.join(evaluation_subdirectory, 'cbf.txt'), 'w') as f:
            f.write(f"CBF for Infections: B1(s) = {x} - Infected Individuals\n")
            f.write(f"CBF for Attendance: B2(s) = {y} - Allowed Students\n")

        return B1, B2

    def verify_forward_invariance(self, B1, B2, allowed_values_over_time, infected_values_over_time,
                                  evaluation_subdirectory):
        """Verify forward invariance using the constructed CBFs."""
        is_invariant = True
        for i in range(len(allowed_values_over_time) - 1):
            s_t = [allowed_values_over_time[i], infected_values_over_time[i]]
            s_t_plus_1 = [allowed_values_over_time[i + 1], infected_values_over_time[i + 1]]

            # Calculate derivatives
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

    def normalize_features(self, features):
        features = np.array(features)  # Convert to numpy array if it's not already
        return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

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

    def construct_lyapunov_function(self, features, evaluation_subdirectory):
        """Construct and save the Lyapunov function based on the learned parameters."""
        # Hyperparameters to adjust
        hidden_dim = 64  # Adjust the number of hidden units
        learning_rate = 0.0001  # Adjust learning rate
        weight_decay = 1e-4  # Adjust L2 regularization
        epochs = 100  # Adjust number of epochs

        # Initialize the neural network
        model = LyapunovNet(input_dim=3, hidden_dim=hidden_dim, output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_function = nn.MSELoss()
        loss_values = []

        # Flatten features
        flattened_features = [self.flatten_features(feature) for feature in features]
        features_tensor = torch.tensor(flattened_features, dtype=torch.float32)
        target_tensor = torch.zeros(len(features_tensor), dtype=torch.float32)

        try:
            for epoch in range(epochs):
                optimizer.zero_grad()
                V_values = model(features_tensor).squeeze()

                loss = loss_function(V_values, target_tensor)

                loss.backward()
                optimizer.step()

                loss_values.append(loss.item())
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

            # Save the final model parameters (theta) and loss
            torch.save(model.state_dict(), os.path.join(evaluation_subdirectory, 'lyapunov_model.pth'))
            with open(os.path.join(evaluation_subdirectory, 'lyapunov_loss_values.txt'), 'w') as f:
                for i, loss_val in enumerate(loss_values):
                    f.write(f"Epoch {i}: Loss = {loss_val}\n")

            return model, None, loss_values  # Return the model (instead of V function) and None for theta
        except Exception as e:
            print(f"Error in construct_lyapunov_function: {e}")
            return None, None, None

    def plot_lyapunov_loss(self, loss_values, evaluation_subdirectory):
        """Plot the loss function of the Lyapunov function training."""
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Lyapunov Function Training Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_subdirectory, 'lyapunov_loss_plot.png'))
        plt.show()

    def plot_lyapunov_function_with_context(self, V, features, method_label, evaluation_subdirectory):
        """Plot Lyapunov function behavior in DFE and EE regions."""
        V_values = []
        delta_V_values = []
        dfe_indices = []
        ee_indices = []

        for i in range(len(features) - 1):
            feature = self.flatten_features(features[i])
            next_feature = self.flatten_features(features[i + 1])

            feature_tensor = torch.tensor(feature, dtype=torch.float32)
            next_feature_tensor = torch.tensor(next_feature, dtype=torch.float32)

            V_t = V(feature_tensor.unsqueeze(0)).item()
            V_t_plus_1 = V(next_feature_tensor.unsqueeze(0)).item()

            V_values.append(V_t)
            delta_V_values.append(V_t_plus_1 - V_t)

            if feature[1] == 0:  # Assuming the second element is the infection count
                dfe_indices.append(i)
            else:
                ee_indices.append(i)

        plt.figure(figsize=(12, 8))
        plt.scatter([V_values[i] for i in dfe_indices], [delta_V_values[i] for i in dfe_indices],
                    color='blue', s=5, alpha=0.5, label='DFE region (Infected = 0)')
        plt.scatter([V_values[i] for i in ee_indices], [delta_V_values[i] for i in ee_indices],
                    color='red', s=5, alpha=0.5, label='EE region (Infected > 0)')
        plt.axhline(y=0, color='black', linestyle='--', label=r'$\Delta V = 0$')

        plt.xlabel('Lyapunov Function Value $V$')
        plt.ylabel(r'$\Delta V$')
        plt.title(f'Lyapunov Function Behavior in DFE and EE Regions\nMethod: {method_label}')
        plt.legend()
        plt.grid(True)

        # Dynamically set axis limits
        v_min, v_max = min(V_values), max(V_values)
        dv_min, dv_max = min(delta_V_values), max(delta_V_values)
        plt.xlim(v_min - 0.1 * (v_max - v_min), v_max + 0.1 * (v_max - v_min))
        plt.ylim(dv_min - 0.1 * (dv_max - dv_min), dv_max + 0.1 * (dv_max - dv_min))

        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_subdirectory, f'lyapunov_function_with_context_{method_label}.png'))
        plt.close()

        # Print some statistics for debugging
        print(f"Total points: {len(V_values)}")
        print(f"DFE points: {len(dfe_indices)}")
        print(f"EE points: {len(ee_indices)}")
        print(f"V value range: [{v_min}, {v_max}]")
        print(f"Delta V value range: [{dv_min}, {dv_max}]")

    def verify_lyapunov_stability(self, V, features, evaluation_subdirectory):
        dfe_stable = True
        ee_stable = True

        def flatten_feature(feature):
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
                V_t = V(feature_tensor.unsqueeze(0))
                V_t_plus_1 = V(next_feature_tensor.unsqueeze(0))
                delta_V = V_t_plus_1 - V_t

                print(f"V_t: {V_t.item()}, V_t_plus_1: {V_t_plus_1.item()}, delta_V: {delta_V.item()}")

                if features[i][1] == 0:  # DFE region
                    if delta_V.item() > 0:
                        dfe_stable = False
                else:  # EE region
                    if delta_V.item() > 0:
                        ee_stable = False
            except Exception as e:
                print(f"Error during V function evaluation: {e}")
                raise

        with open(os.path.join(evaluation_subdirectory, 'lyapunov_verification.txt'), 'w') as f:
            if dfe_stable:
                f.write("The system is stochastically stable in the Disease-Free Equilibrium (DFE) region.\n")
            else:
                f.write("The system is not stochastically stable in the Disease-Free Equilibrium (DFE) region.\n")

            if ee_stable:
                f.write("The system is stochastically stable in the Endemic Equilibrium (EE) region.\n")
            else:
                f.write("The system is not stochastically stable in the Endemic Equilibrium (EE) region.\n")

        return dfe_stable, ee_stable

    def evaluate(self, run_name, num_episodes=1, alpha=0.5, csv_path=None, x=10, y=50, z=90):
        model_subdirectory = os.path.join(self.model_directory, self.agent_type, run_name)
        model_file_path = os.path.join(model_subdirectory, f'model.pt')
        results_directory = self.results_subdirectory

        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found in {model_file_path}")

        self.model.load_state_dict(torch.load(model_file_path))
        self.model.eval()
        print(f"Loaded model from {model_file_path}")

        if csv_path:
            self.community_risk_values = self.read_community_risk_from_csv(csv_path)
            self.max_weeks = len(self.community_risk_values)
            print(f"Community Risk Values: {self.community_risk_values}")

        total_rewards = []
        evaluation_subdirectory = os.path.join(results_directory, run_name)
        os.makedirs(evaluation_subdirectory, exist_ok=True)
        csv_file_path = os.path.join(evaluation_subdirectory, f'evaluation_metrics_{run_name}.csv')

        allowed_values_over_time = []
        infected_values_over_time = []

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Step', 'State', 'Action', 'Community Risk', 'Total Reward'])

            for episode in range(num_episodes):
                initial_infection_count = 20
                initial_community_risk = self.community_risk_values[0] if csv_path else 0
                state = np.array([initial_infection_count, int(initial_community_risk * 100)], dtype=np.float32)
                print(f"Initial state for episode {episode + 1}: {state}")
                total_reward = 0
                done = False
                step = 0

                while not done:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = self.model(state_tensor)
                        actions = q_values.argmax(dim=1).cpu().numpy()
                        actions = [action * 50 for action in actions]

                    next_state, reward, done, _, info = self.env.step((actions, alpha))
                    next_state = np.array(next_state, dtype=np.float32)
                    total_reward += reward

                    writer.writerow(
                        [episode + 1, step + 1, int(state[0]), actions[0], info['community_risk'], reward]
                    )

                    if episode == 0:
                        allowed_values_over_time.append(actions[0])
                        infected_values_over_time.append(next_state[0])

                    state = next_state
                    step += 1

                total_rewards.append(total_reward)

                print(f"Episode {episode + 1}: Total Reward = {total_reward}")

            avg_reward = np.mean(total_rewards)
        print(f"Cumulative Reward over {num_episodes} episodes: {sum(total_rewards)}")

        min_length = min(len(allowed_values_over_time), len(infected_values_over_time), len(self.community_risk_values))
        allowed_values_over_time = allowed_values_over_time[:min_length]
        infected_values_over_time = infected_values_over_time[:min_length]
        community_risk_values = self.community_risk_values[:min_length]
        weeks = range(1, min_length + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        sns.lineplot(x=weeks, y=community_risk_values,
                     marker='s', linestyle='--', color='darkgreen', linewidth=2.5, ax=ax1)
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Community Risk', color='darkgreen')
        ax1.tick_params(axis='y', labelcolor='darkgreen')

        ax2.bar(weeks, allowed_values_over_time,
                color='blue', alpha=0.6, width=0.4, align='center', label='Allowed')
        ax2.bar(weeks, infected_values_over_time,
                color='red', alpha=0.6, width=0.4, align='edge', label='Infected')
        ax2.set_xlabel('Week')
        ax2.set_ylabel('Allowed and Infected Values')
        ax2.legend(loc='upper left')

        ticks = range(0, min_length, max(1, min_length // 10))
        labels = [f'Week {i + 1}' for i in ticks]

        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels, rotation=45)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(labels, rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_subdirectory, f"evaluation_plot_{run_name}.png"))
        plt.show()

        # Identification of Safety Set
        infection_safe, attendance_safe = self.identify_safety_set(
            allowed_values_over_time, infected_values_over_time, x, y, z, evaluation_subdirectory
        )

        # Construction of Control Barrier Functions
        B1, B2 = self.construct_cbf(allowed_values_over_time, infected_values_over_time, evaluation_subdirectory, x, y)

        # Verification of Forward Invariance
        is_invariant = self.verify_forward_invariance(B1, B2, allowed_values_over_time, infected_values_over_time,
                                                      evaluation_subdirectory)

        # Lyapunov Function and Stability Analysis
        features = self.extract_features(allowed_values_over_time, infected_values_over_time)
        V, theta, loss_values = self.construct_lyapunov_function(features, evaluation_subdirectory)
        self.plot_lyapunov_loss(loss_values, evaluation_subdirectory)
        dfe_stable, ee_stable = self.verify_lyapunov_stability(V, features, evaluation_subdirectory)
        self.plot_lyapunov_function_with_context(V, features, "DQN Evaluation", evaluation_subdirectory)

        # Save final results
        with open(os.path.join(evaluation_subdirectory, 'final_results.txt'), 'w') as f:
            f.write(f"Cumulative Reward over {num_episodes} episodes: {sum(total_rewards)}\n")
            f.write(f"Average Reward: {avg_reward}\n")
            f.write(f"Safety Set Invariance for Infections <= {x}: {'Passed' if infection_safe else 'Failed'}\n")
            f.write(f"Safety Set Invariance for Attendance <= {y}: {'Passed' if attendance_safe else 'Failed'}\n")
            f.write(f"Forward Invariance Verified: {'Yes' if is_invariant else 'No'}\n")
            f.write(f"Stochastic Stability in DFE region: {'Stable' if dfe_stable else 'Unstable'}\n")
            f.write(f"Stochastic Stability in EE region: {'Stable' if ee_stable else 'Unstable'}\n")

        return avg_reward


def load_saved_model(model_directory, agent_type, run_name, input_dim, hidden_dim, action_space_nvec):
    """
    Load a saved DeepQNetwork model from the subdirectory.

    Args:
    model_directory: Base directory where models are stored.
    agent_type: Type of the agent, used in directory naming.
    run_name: Name of the run, used in directory naming.
    input_dim: Input dimension of the model.
    hidden_dim: Hidden layer dimension.
    action_space_nvec: Action space vector size.

    Returns:
    model: The loaded DeepQNetwork model, or None if loading failed.
    """
    # Construct the model subdirectory path
    model_subdirectory = os.path.join(model_directory, agent_type, run_name)

    # Construct the model file path
    model_file_path = os.path.join(model_subdirectory, 'model.pt')

    # Check if the model file exists
    if not os.path.exists(model_file_path):
        print(f"Model file not found in {model_file_path}")
        return None

    # Initialize a new model instance
    model = DeepQNetwork(input_dim, hidden_dim, action_space_nvec)

    # Load the saved model state into the model instance
    model.load_state_dict(torch.load(model_file_path))
    model.eval()  # Set the model to evaluation mode

    return model
def calculate_explained_variance(actual_rewards, predicted_rewards):
    actual_rewards = np.array(actual_rewards)
    predicted_rewards = np.array(predicted_rewards)
    variance_actual = np.var(actual_rewards, ddof=1)
    variance_unexplained = np.var(actual_rewards - predicted_rewards, ddof=1)
    explained_variance = 1 - (variance_unexplained / variance_actual)
    return explained_variance


def visualize_explained_variance(explained_variance_per_episode, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(explained_variance_per_episode, label='Explained Variance')
    plt.xlabel('Episode')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def eval_with_csv(self, alpha, episodes, csv_path):
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
    if not os.path.isfile(eval_file_path):
        with open(eval_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Alpha', 'Episode', 'Step', 'Infections', 'Allowed', 'Reward', 'CommunityRisk'])

    for episode in tqdm(range(episodes)):
        state, _ = self.env.reset()
        c_state = state if isinstance(state, (list, np.ndarray)) else [state]
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

            action = np.argmax(self.q_table[state_idx])

            list_action = list(eval(self.all_actions[action]))
            c_list_action = [i * 50 for i in list_action]

            action_alpha_list = [*c_list_action, alpha]
            self.env.campus_state.community_risk = comm_risk

            next_state, reward, terminated, _, info = self.env.step(action_alpha_list)
            eps_rewards.append(reward)
            infected.append(info['infected'])
            allowed.append(info['allowed'])
            community_risk.append(info['community_risk'])
            episode_infections += sum(info['infected'])

            if last_action is not None and last_action != action:
                policy_changes += 1
                last_action = action

            total_class_capacity_utilized += sum(info['allowed'])

            c_state = next_state

        infected_dict[episode] = infected
        allowed_dict[episode] = allowed
        rewards_dict[episode] = eps_rewards
        community_risk_dict[episode] = community_risk

    print("infected: ", infected_dict, "allowed: ", allowed_dict, "rewards: ", rewards_dict, "community_risk: ",
              community_risk_dict)
    for episode in infected_dict:
        plt.figure(figsize=(15, 5))
        infections = [inf[0] for inf in infected_dict[episode]] if episode in infected_dict else []
        allowed_students = [alw[0] for alw in allowed_dict[episode]] if episode in allowed_dict else []
        rewards = rewards_dict[episode] if episode in rewards_dict else []
        community_risk = community_risk_dict[episode] if episode in community_risk_dict else []

        steps = np.arange(len(infections))

        bar_width = 0.4
        offset = bar_width / 4

        plt.bar(steps - offset, infections, width=bar_width, label='Infections', color='#bc5090', align='center')
        plt.bar(steps + offset, allowed_students, width=bar_width, label='Allowed Students', color='#003f5c',
                    alpha=0.5, align='edge')
        plt.plot(steps, rewards, label='Rewards', color='#ffa600', linestyle='-', marker='o')

        plt.xlabel('Step')
        plt.ylabel('Count')
        plt.title(f'Evaluation of agent {self.run_name} Policy for {episode} episodes')
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(eval_dir, f'{self.run_name}_metrics.png')
        plt.savefig(fig_path)
        print(f"Figure saved to {fig_path}")

        plt.close()

    with open(eval_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
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
