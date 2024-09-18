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
from .utilities import load_config, ExplorationRateDecay
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
from scipy.signal import argrelextrema
from io import StringIO

epsilon = 1e-10
SEED = 100




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
set_seed(SEED)  # Replace 42 with your desired seed value
allowed = torch.tensor([0, 50, 100])  # Updated allowed values

def log_all_states_visualizations(q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory):
    file_paths = visualize_all_states(q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory, 100)

def log_states_visited(states, visit_counts, alpha, results_subdirectory):
    file_paths = states_visited_viz(states, visit_counts, alpha, results_subdirectory)

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


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(ActorCriticNetwork, self).__init__()

        num_layers = 18  # Number of hidden layers

        # Create a list to hold the layers for the encoder
        encoder_layers = []

        # Add the first layer with input_dim
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers.append(nn.ReLU())

        # Add additional hidden layers
        for _ in range(num_layers - 1):  # num_layers includes the first layer
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())

        # Use nn.Sequential to define the encoder with the specified number of layers
        self.encoder = nn.Sequential(*encoder_layers)

        # Define the policy head and value head
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        encoded = self.encoder(x)
        policy_logits = self.policy_head(encoded)
        value = self.value_head(encoded)
        return policy_logits, value


class A2CCustomAgent:
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
        self.agent_type = "a2c_custom"
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
        # wandb.init(project=self.agent_type, name=self.run_name)
        self.env = env

        # Initialize the neural network
        self.input_dim = len(env.reset()[0])
        self.output_dim = env.action_space.nvec[0]
        self.hidden_dim = self.agent_config['agent']['hidden_units']
        self.num_courses = self.env.action_space.nvec[0]

        self.model = ActorCriticNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.agent_config['agent']['learning_rate'])

        self.entropy_coeff = self.agent_config['agent']['ent_coef']
        self.value_loss_coeff = self.agent_config['agent']['vf_coef']

        # Initialize agent-specific configurations and variables

        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']
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
            # print(f"Community Risk Values: {self.community_risk_values}")
        else:
            self.community_risk_values = None
            self.max_weeks = self.env.campus_state.model.get_max_weeks()
        self.save_path = self.results_subdirectory  # Ensure save path is set
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def get_next_state(self, current_state, alpha=0.2):
        current_infected = current_state[0].unsqueeze(0)
        community_risk = current_state[1].unsqueeze(0)

        label, allowed_value, new_infected, _, _, _ = self.get_label(current_infected, community_risk, alpha)

        return torch.tensor([new_infected.item(), community_risk.item()], dtype=torch.float32)

    def generate_diverse_states(self, num_samples):
        rng = np.random.default_rng(SEED)  # Create a new RNG with the seed
        infected = rng.uniform(0, 100, num_samples)
        risk = rng.uniform(0, 1, num_samples)
        return torch.tensor(np.column_stack((infected, risk)), dtype=torch.float32)
    def read_community_risk_from_csv(self, csv_path):
        try:
            community_risk_df = pd.read_csv(csv_path)
            return community_risk_df['Risk-Level'].tolist()
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy_logits, _ = self.model(state_tensor)

            # Apply softmax with temperature
            temperature = 0.3
            policy_logits = policy_logits / temperature

            # Ensure probabilities sum to 1 and are non-negative
            logits_max = policy_logits.max(dim=-1, keepdim=True).values
            probs = F.softmax(policy_logits - logits_max, dim=-1)

            # Add small epsilon to avoid zero probabilities
            epsilon = 1e-6
            probs = probs + epsilon
            probs = probs / probs.sum()  # Renormalize

            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                print(f"Warning: Invalid probability distribution: {probs}")
                action_index = torch.argmax(policy_logits).item()  # Choose the action with highest logit
            else:
                action_index = torch.multinomial(probs, 1).item()

            scaled_action = self.scale_action(action_index, self.output_dim)
            return [scaled_action]

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
        # q_values = q_values.to(self.device)

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

    def scale_action(self, action_index, num_actions):
        """
        Scale the action index to the corresponding allowed value.
        E.g., action_index=0 -> 0, action_index=1 -> 50, action_index=2 -> 100
        """
        max_value = 100
        step_size = max_value / (num_actions - 1)
        return int(action_index * step_size)

    def reverse_scale_action(self, action, num_actions):
        """
        Reverse scale the action value back to an action index.
        E.g., action=50 -> 1, action=100 -> 2
        """
        max_value = 100
        step_size = max_value / (num_actions - 1)

        # Check if the action is a list/array and get the first element
        if isinstance(action, (list, np.ndarray)):
            action = action[0]

        return round(action / step_size)

    def save_model(self, run_name):
        """Save the trained model's state dict."""
        model_subdirectory = os.path.join(self.model_directory, self.agent_type, run_name)
        if not os.path.exists(model_subdirectory):
            os.makedirs(model_subdirectory, exist_ok=True)
        model_file_path = os.path.join(model_subdirectory, 'model.pt')
        torch.save(self.model.state_dict(), model_file_path)
        print(f"Model saved at {model_file_path}")

    def load_model(self, run_name):
        """Load the model's state dict from the saved file."""
        model_subdirectory = os.path.join(self.model_directory, self.agent_type, run_name)
        model_file_path = os.path.join(model_subdirectory, 'model.pt')
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found in {model_file_path}")

        self.model.load_state_dict(torch.load(model_file_path))
        self.model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {model_file_path}")

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
                                fieldnames=['episode', 'cumulative_reward', 'average_reward',
                                            'convergence_rate', 'sample_efficiency', 'policy_entropy',
                                            'space_complexity'])
        if not file_exists:
            writer.writeheader()

        for episode in range(self.max_episodes):
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            done = False
            episode_rewards = []
            log_probs = []
            values = []
            rewards = []
            entropies = []

            # Episode-specific tracking
            episode_allowed = []
            episode_infected = []
            visited_states = set()

            while not done:
                # Select action (returns scaled action)
                action = self.select_action(state)
                # print("action: ", action)

                # Reverse scale action before applying it
                original_action = self.reverse_scale_action(action, self.output_dim)  # This will map 50 -> 1, etc.
                # print('original_action: ', original_action)
                next_state, reward, done, _, info = self.env.step((action, alpha))
                next_state = np.array(next_state, dtype=np.float32)

                if np.isnan(next_state).any() or np.isinf(next_state).any():
                    print(f"NaN or Inf detected in next_state: {next_state}")
                    next_state = np.nan_to_num(next_state, nan=0.0, posinf=1e6, neginf=-1e6)

                # Continue with the rest of the code...

                # Continue with the rest of the code
                policy_logits, value = self.model(torch.FloatTensor(state).unsqueeze(0))
                policy_dist = F.softmax(policy_logits, dim=-1)
                log_prob = torch.log(policy_dist.squeeze(0)[original_action])
                entropy = -torch.sum(policy_dist * torch.log(policy_dist), dim=-1)

                # Store episode data
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)

                # Track allowed and infected counts
                episode_allowed.append(sum(info.get('allowed', [])))  # Sum all courses' allowed students
                episode_infected.append(sum(info.get('infected', [])))

                # Update state and track visited states
                state = next_state
                state_tuple = tuple(state)
                visited_states.add(state_tuple)
                visited_state_counts[state_tuple] = visited_state_counts.get(state_tuple, 0) + 1

            # Calculate the n-step returns and losses after the episode
            R = 0  # Terminal state has no future value
            returns = []
            policy_loss = 0
            value_loss = 0

            for r in reversed(rewards):
                R = r
                returns.insert(0, R)

            returns = torch.FloatTensor(returns)
            values = torch.cat(values).squeeze()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for log_prob, value, entropy, advantage, R in zip(log_probs, values, entropies, advantages, returns):
                policy_loss -= log_prob * advantage.detach()  # Policy loss
                value_loss += F.mse_loss(value, R)  # Value loss
                entropy_loss = -entropy.mean()  # Entropy loss

            # Combine losses
            entropy_loss = -torch.stack(entropies).mean()
            total_loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy_loss

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)
            # Store episode rewards
            allowed_means_per_episode.append(np.mean(episode_allowed))
            infected_means_per_episode.append(np.mean(episode_infected))
            actual_rewards.append(sum(rewards))
            predicted_rewards.append(values.detach().numpy().tolist())

            # Logging metrics to CSV
            avg_episode_return = np.mean(rewards)
            cumulative_reward = sum(rewards)
            sample_efficiency = len(visited_states)  # Unique states visited in the episode

            metrics = {
                'episode': episode,
                'cumulative_reward': cumulative_reward,
                'average_reward': avg_episode_return,
                'sample_efficiency': sample_efficiency,
                'policy_entropy': -entropy_loss.item(),
                'space_complexity': len(visited_state_counts)
            }
            writer.writerow(metrics)

            pbar.update(1)
            pbar.set_description(f"Total Reward: {cumulative_reward:.2f}, Epsilon: {self.exploration_rate:.2f}")

        pbar.close()
        csvfile.close()

        # Plot the training metrics
        self.plot_metrics(open(self.csv_file_path, 'r').read())

        # Log and save the allowed and infected means per episode
        mean_allowed = round(np.mean(allowed_means_per_episode))
        mean_infected = round(np.mean(infected_means_per_episode))
        summary_file_path = os.path.join(self.results_subdirectory, 'mean_allowed_infected.csv')

        with open(summary_file_path, 'w', newline='') as summary_csvfile:
            summary_writer = csv.DictWriter(summary_csvfile, fieldnames=['mean_allowed', 'mean_infected'])
            summary_writer.writeheader()
            summary_writer.writerow({'mean_allowed': mean_allowed, 'mean_infected': mean_infected})
        self.save_model(self.run_name)

        return self.model

    def plot_metrics(self, csv_data):
        # Convert the CSV string to a DataFrame using StringIO
        df = pd.read_csv(StringIO(csv_data))

        # Define metrics to plot
        metrics = ['cumulative_reward', 'average_reward', 'sample_efficiency', 'policy_entropy',
                   'space_complexity']

        # Plot each metric separately
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(df['episode'], df[metric])
            plt.title(metric.replace('_', ' ').title())
            plt.xlabel('Episode')
            plt.ylabel('Value')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_subdirectory, f'{metric}.png'))

    def generate_all_states(self):
        value_range = range(0, 101, 10)  # Define the range of values for community risk and infected students
        input_dim = self.model.encoder[0].in_features  # The input dimension expected by the model

        if input_dim == 2:
            # If the model expects only 2 inputs, we'll use the first course and community risk
            all_states = [np.array([i, j]) for i in value_range for j in value_range]
        else:
            # Generate states for all courses and community risk
            course_combinations = itertools.product(value_range, repeat=self.num_courses)
            all_states = [np.array(list(combo) + [risk]) for combo in course_combinations for risk in value_range]

            # Ensure that all states match the input dimension of the model
            all_states = [state[:input_dim] if len(state) > input_dim else
                          np.pad(state, (0, max(0, input_dim - len(state))), 'constant')
                          for state in all_states]

        # Convert states to a consistent format (e.g., list of lists or list of arrays)
        all_states = [list(state) for state in all_states]  # Ensure all states are lists, not tuples or arrays

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

        # Smoothing the curve
        central_tendency_smooth = self.moving_average(central_tendency, 100)
        lower_bounds_smooth = self.moving_average(lower_bounds, 100)
        upper_bounds_smooth = self.moving_average(upper_bounds, 100)
        episodes_smooth = list(range(len(central_tendency_smooth)))

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



        # Smoothing the curve
        means_smooth = self.moving_average(means, 100)
        lower_bounds_smooth = self.moving_average(lower_bounds, 100)
        upper_bounds_smooth = self.moving_average(upper_bounds, 100)
        episodes_smooth = list(range(len(means_smooth)))


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
    def identify_safety_set(self, allowed_values_over_time, infected_values_over_time, x, y, z,
                            evaluation_subdirectory):
        """Identify and plot the safety set based on fixed constraints."""
        # Process infected_values_over_time to ensure it contains only scalar values
        processed_infected_values = [
            infected[0] if isinstance(infected, (list, tuple)) else infected
            for infected in infected_values_over_time
        ]

        # 1. Ensure that no more than `x` infected individuals are present for more than `z%` of the time.
        time_exceeding_x = sum(1 for val in processed_infected_values if val > x)
        time_within_x = len(processed_infected_values) - time_exceeding_x
        infection_safety_percentage = (time_within_x / len(processed_infected_values)) * 100

        # 2. Ensure that `y` allowed students are present at least `z%` of the time.
        time_with_y_present = sum(1 for val in allowed_values_over_time if val >= y)
        attendance_safety_percentage = (time_with_y_present / len(allowed_values_over_time)) * 100

        with open(os.path.join(evaluation_subdirectory, 'safety_set.txt'), 'w') as f:
            f.write(f"Safety Condition: No more than {x} infections for {100 - z}% of time: "
                    f"{infection_safety_percentage}%\n")
            f.write(f"Safety Condition: At least {y} allowed students for {z}% of time: "
                    f"{attendance_safety_percentage}%\n")

        plt.figure(figsize=(10, 6))
        plt.scatter(allowed_values_over_time, processed_infected_values, color='blue', label='State Points')
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

    def extract_features(self, infected_values_over_time, community_risk_values):
        """Extract features for constructing the Lyapunov function using infected and community risk."""
        features = []
        for infected, risk in zip(infected_values_over_time, community_risk_values):
            features.append([infected, risk])
        return features

    def normalize_features(self, features):
        features = np.array(features)  # Convert to numpy array if it's not already
        return (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    def flatten_features(self, feature):
        if isinstance(feature, (list, tuple)):
            flattened_feature = [item for sublist in feature for item in
                                 (sublist if isinstance(sublist, (list, tuple)) else [sublist])]
        else:
            flattened_feature = [feature]

        # Ensure we only take the first two elements
        return flattened_feature[:2]

    def construct_lyapunov_function(self, infected_values, community_risk_values, alpha):
        model = LyapunovNet(input_dim=2, hidden_dim=16, output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.000009)
        loss_values = []
        epochs = 1000
        epsilon = 1e-8

        # Combine the infected values and community risk values to form the state tensor
        states = np.column_stack((infected_values[:-1], community_risk_values[:-1]))  # Current states
        next_states = np.column_stack((infected_values[1:], community_risk_values[1:]))  # Next states (shifted)

        states_tensor = torch.tensor(states, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            V = model(states_tensor)

            # Use the actual next states observed over time
            V_next = model(next_states_tensor)

            # Lyapunov stability conditions
            positive_definite_loss = F.relu(-V + epsilon).mean()  # Ensure Lyapunov is non-negative
            decreasing_loss = F.relu(V_next - V + epsilon).mean()  # Ensure Lyapunov decreases over time

            loss = positive_definite_loss + decreasing_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf detected at epoch {epoch}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_values.append(loss.item())

        return model, loss_values
    def evaluate_lyapunov(self, model, features, alpha):
        # Evaluate using the community risk and infected values from the CSV
        eval_states = torch.tensor([[f[0], f[1]] for f in features], dtype=torch.float32)



        with torch.no_grad():
            V = model(eval_states).squeeze()
            V_next = model(self.get_next_states(eval_states, alpha)).squeeze()

            positive_definite = (V > 0).float().mean()
            decreasing = (V_next < V).float().mean()

        print(f"Positive definite: {positive_definite.item():.2%}")
        print(f"Decreasing: {decreasing.item():.2%}")

        return positive_definite.item(), decreasing.item()

    def plot_loss_function(self, loss_values, alpha, run_name):
        if not loss_values:
            print("No loss values to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Lyapunov Function Training Loss (Run: {run_name}, Alpha: {alpha})')
        plt.yscale('log')  # Use log scale for y-axis
        plt.grid(True)
        plt.tight_layout()

        # Add text with min and max loss values
        min_loss = min(loss_values)
        max_loss = max(loss_values)
        plt.text(0.05, 0.95, f'Min Loss: {min_loss:.6f}\nMax Loss: {max_loss:.6f}',
                 transform=plt.gca().transAxes, verticalalignment='top')

        plt.savefig(os.path.join(self.save_path, f'lyapunov_loss_plot_{run_name}_alpha_{alpha}.png'))
        plt.close()

        print(f"Min Loss: {min_loss}, Max Loss: {max_loss}")

    def verify_lyapunov_stability(self, model, features, alpha):
        # For evaluation, you will be using the community risk data from the CSV
        positive_definite, decreasing = self.evaluate_lyapunov(model, features, alpha)

        with open(os.path.join(self.save_path, 'lyapunov_verification.txt'), 'w') as f:
            f.write(f"Lyapunov Function Evaluation:\n")
            f.write(f"Positive definite: {positive_definite:.2%}\n")
            f.write(f"Decreasing: {decreasing:.2%}\n")

            if positive_definite > 0.99 and decreasing > 0.99:
                f.write("The Lyapunov function satisfies stability conditions.\n")
            else:
                f.write("The Lyapunov function does not fully satisfy stability conditions.\n")

        return positive_definite > 0.99 and decreasing > 0.99

    def get_next_states(self, states, alpha):
        next_states = []
        for state in states:

            with torch.no_grad():
                # q_values = self.model(state.unsqueeze(0))
                # action = q_values.argmax(dim=1).item()
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                policy_logits, _ = self.model(state_tensor)
                policy_dist = F.softmax(policy_logits, dim=-1)
                action_index = torch.multinomial(policy_dist, 1).item()
                scaled_action = self.scale_action(action_index, self.output_dim)
                allowed_value = scaled_action * 50  # Convert to allowed values (0, 50, 100)

            current_infected = state[0].item()
            community_risk = state[1].item()

            alpha_infection = 0.005
            beta_infection = 0.01

            new_infected = ((alpha_infection * current_infected) * allowed_value) + \
                           ((beta_infection * community_risk) * allowed_value ** 2)

            new_infected = min(new_infected, allowed_value)
            next_states.append(torch.tensor([new_infected, community_risk], dtype=torch.float32))

        return torch.stack(next_states)

    def plot_steady_state_and_stable_points(self, lyapunov_model, run_name, alpha, infected_values,
                                            community_risk_values,
                                            num_simulations=1000, num_steps=100):
        # Ensure that the infected and risk values are of the same length
        assert len(infected_values) == len(
            community_risk_values), "Infected values and risk values must have the same length."

        # Convert infected and risk values to numpy arrays for easy manipulation
        infected = np.array(infected_values)
        risk = np.array(community_risk_values)

        # Create a grid using the infected and risk values from the simulation data
        X, Y = np.meshgrid(np.linspace(min(infected), max(infected), 100),  # Create a uniform grid for plotting
                           np.linspace(min(risk), max(risk), 100))

        # Combine the grid for calculating the Lyapunov function values
        grid_states = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)

        # Calculate Lyapunov function values for the grid
        with torch.no_grad():
            V_values = lyapunov_model(grid_states).squeeze().cpu().numpy().reshape(X.shape)

        # Calculate Lyapunov function change (ΔV)
        next_states = self.get_next_states(grid_states, alpha)
        with torch.no_grad():
            V_next_values = lyapunov_model(next_states).squeeze().cpu().numpy().reshape(X.shape)

        # Ensure that delta_V is calculated as the difference between consecutive states' Lyapunov values
        delta_V = V_next_values - V_values
        print(f"Delta V: {delta_V.min()}, {delta_V.max()}")
        print("Delta V values:", delta_V)

        # Calculate metrics
        positive_definite_percentage = np.mean(V_values > 0) * 100  # Proportion of positive definite values
        mean_lyapunov_value = np.mean(V_values)  # Mean Lyapunov value
        mean_delta_V = np.mean(delta_V)  # Average rate of change (Delta V)
        time_to_equilibrium = np.argmax(np.abs(delta_V) < 1e-3) if np.any(np.abs(delta_V) < 1e-3) else len(
            delta_V)  # Time to equilibrium
        stable_region_percentage = np.mean(np.abs(V_values) < 1e-3) * 100  # Percentage of time in stable region
        variance_lyapunov = np.var(V_values)  # Variance of Lyapunov values

        # Simulate steady-state distribution using infected_values and community_risk_values
        steady_state_samples = list(zip(infected_values, community_risk_values))
        hist, xedges, yedges = np.histogram2d(
            [s[0] for s in steady_state_samples],
            [s[1] for s in steady_state_samples],
            bins=[100, 100], range=[[min(infected_values), max(infected_values)],
                                    [min(community_risk_values), max(community_risk_values)]]
        )
        hist = hist / hist.sum()  # Normalize to get the probability distribution

        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

        # Plot 1: Steady-State Distribution using simulated infected and risk values
        c1 = ax1.imshow(hist.T, extent=[min(infected_values), max(infected_values), min(community_risk_values),
                                        max(community_risk_values)], origin='lower', aspect='auto', cmap='gray_r')
        ax1.set_title(f'Simulated Steady-State Distribution (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('Infected')
        ax1.set_ylabel('Community Risk')
        fig.colorbar(c1, ax=ax1, label='Probability')

        # Plot 2: Lyapunov Function
        c2 = ax2.contourf(X, Y, V_values, levels=20, cmap='gray')
        ax2.set_title(f'Lyapunov Function (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('Infected')
        ax2.set_ylabel('Community Risk')
        fig.colorbar(c2, ax=ax2, label='Lyapunov Value')

        # Plot 3: Lyapunov Function Change (ΔV)
        c3 = ax3.contourf(X, Y, delta_V, levels=20, cmap='gray', center=0)
        ax3.set_title(f'Lyapunov Function Change (Run: {run_name}, Alpha: {alpha})')
        ax3.set_xlabel('Infected')
        ax3.set_ylabel('Community Risk')
        fig.colorbar(c3, ax=ax3, label='Lyapunov Value Change')

        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'steady_state_and_stable_points_{run_name}_alpha_{alpha}.png'))
        plt.close()

        # Save the metrics to a CSV file
        metrics_file_path = os.path.join(self.save_path, f'lyapunov_metrics_{run_name}_alpha_{alpha}.csv')
        with open(metrics_file_path, mode='w', newline='') as metrics_file:
            writer = csv.writer(metrics_file)
            # Write headers
            writer.writerow([
                'Run Name', 'Alpha', 'Mean Lyapunov Value', 'Proportion Positive Definite (%)',
                'Average Rate of Change (Delta V)', 'Time to Equilibrium (steps)',
                'Percentage in Stable Region (%)', 'Variance of Lyapunov Values'
            ])
            # Write the metrics
            writer.writerow([
                run_name, alpha, mean_lyapunov_value, positive_definite_percentage, mean_delta_V,
                time_to_equilibrium, stable_region_percentage, variance_lyapunov
            ])

        print(f"Metrics saved to {metrics_file_path}")

    def plot_lyapunov_change(self, V, features, run_name, alpha):
        infected = np.linspace(0, 100, 50)
        risk = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(infected, risk)
        states = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)

        with torch.no_grad():
            V_values = V(states).squeeze().cpu().numpy().reshape(X.shape)
            next_states = self.get_next_states(states, alpha)
            V_next_values = V(next_states).squeeze().cpu().numpy().reshape(X.shape)
            delta_V = V_next_values - V_values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        im1 = ax1.imshow(V_values, extent=[0, 100, 0, 1], origin='lower', aspect='auto', cmap='plasma')
        ax1.set_title(f'Lyapunov Function V(x) (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('Infected')
        ax1.set_ylabel('Community Risk')
        fig.colorbar(im1, ax=ax1, label='V(x)')

        im2 = ax2.imshow(delta_V, extent=[0, 100, 0, 1], origin='lower', aspect='auto', cmap='plasma')
        ax2.set_title(f'Change in Lyapunov Function ΔV(x) (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('Infected')
        ax2.set_ylabel('Community Risk')
        fig.colorbar(im2, ax=ax2, label='ΔV(x)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'lyapunov_change_{run_name}_alpha_{alpha}.png'))
        plt.close()

    def plot_lyapunov_properties(self, V, infected_values, community_risk_values, run_name, alpha):
        # Convert infected and community risk values into tensors
        infected_tensor = torch.tensor(infected_values, dtype=torch.float32)
        risk_tensor = torch.tensor(community_risk_values, dtype=torch.float32)

        # Stack them together to form the state representation
        states_tensor = torch.stack([infected_tensor, risk_tensor], dim=1)

        # Compute V(x) values and their differences (ΔV)
        with torch.no_grad():
            V_values = V(states_tensor).squeeze()  # Compute Lyapunov values for the states
            delta_V = V_values[1:] - V_values[:-1]  # Compute differences between consecutive values

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

        # Plot V(x) for each state index
        ax1.plot(V_values.cpu().numpy(), label='V(x)')
        ax1.axhline(y=0, color='r', linestyle='--', label='V(x) = 0')
        ax1.set_title(f'Lyapunov Function (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('State Index')
        ax1.set_ylabel('V(x)')
        ax1.legend()
        ax1.grid(True)

        # Plot ΔV(x) for each state index
        ax2.plot(delta_V.cpu().numpy(), label='ΔV(x)')
        ax2.axhline(y=0, color='r', linestyle='--', label='ΔV(x) = 0')
        ax2.set_title(f'Change in Lyapunov Function (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('State Index')
        ax2.set_ylabel('ΔV(x)')
        ax2.legend()
        ax2.grid(True)

        # Plot V(x) values as a scatter plot where the color represents the V(x) values
        scatter = ax3.scatter(infected_tensor.cpu().numpy(), risk_tensor.cpu().numpy(),
                              c=V_values.cpu().numpy(), cmap='gray', alpha=0.6)
        ax3.set_title(f'Lyapunov Function (V(x)) vs Infected and Community Risk (Run: {run_name}, Alpha: {alpha})')
        ax3.set_xlabel('Infected')
        ax3.set_ylabel('Community Risk')
        fig.colorbar(scatter, ax=ax3, label='V(x)')
        ax3.grid(True)

        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'lyapunov_properties_{run_name}_alpha_{alpha}.png'))
        plt.close()

        return V_values.mean().item(), delta_V.mean().item()

    def plot_equilibrium_points(self, features, run_name, alpha):
        infected = [f[0] for f in features]
        risk = [f[1] for f in features]

        plt.figure(figsize=(10, 8))
        plt.scatter(risk, infected, c=infected, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Number of Infected')
        plt.xlabel('Community Risk')
        plt.ylabel('Number of Infected')
        plt.title(f'Equilibrium Points: DFE and EE Regions (Run: {run_name}, Alpha: {alpha})')

        # Define DFE and EE regions
        plt.axhline(y=0.5, color='r', linestyle='--', label='DFE/EE Boundary')
        plt.text(0.5, 0.25, 'DFE Region', horizontalalignment='center')
        plt.text(0.5, 0.75, 'EE Region', horizontalalignment='center')

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'equilibrium_points_{run_name}_alpha_{alpha}.png'))
        plt.close()

    # def plot_lyapunov_change(self, V, features, run_name, alpha):
    #     infected = np.linspace(0, 100, 50)
    #     risk = np.linspace(0, 1, 50)
    #     X, Y = np.meshgrid(infected, risk)
    #     states = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)
    #
    #     with torch.no_grad():
    #         V_values = V(states).squeeze().cpu().numpy().reshape(X.shape)
    #         next_states = self.get_next_states(states, alpha)
    #         V_next_values = V(next_states).squeeze().cpu().numpy().reshape(X.shape)
    #         delta_V = V_next_values - V_values
    #
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    #
    #     im1 = ax1.imshow(V_values, extent=[0, 100, 0, 1], origin='lower', aspect='auto', cmap='viridis')
    #     ax1.set_title(f'Lyapunov Function V(x) (Run: {run_name}, Alpha: {alpha})')
    #     ax1.set_xlabel('Infected')
    #     ax1.set_ylabel('Community Risk')
    #     fig.colorbar(im1, ax=ax1, label='V(x)')
    #
    #     im2 = ax2.imshow(delta_V, extent=[0, 100, 0, 1], origin='lower', aspect='auto', cmap='coolwarm')
    #     ax2.set_title(f'Change in Lyapunov Function ΔV(x) (Run: {run_name}, Alpha: {alpha})')
    #     ax2.set_xlabel('Infected')
    #     ax2.set_ylabel('Community Risk')
    #     fig.colorbar(im2, ax=ax2, label='ΔV(x)')
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.save_path, f'lyapunov_change_{run_name}_alpha_{alpha}.png'))
    #     plt.close()



    def analyze_markov_chain_stability(self, states, next_states, run_name):
        """Analyze the Markov Chain stability by computing the transition matrix and stationary distribution."""
        # Convert states and next_states to tuples for consistency
        states = [tuple(state) for state in states]
        next_states = [tuple(next_state) for next_state in next_states]

        # Identify unique states and create a mapping to indices
        unique_states = sorted(set(states + next_states))
        state_to_index = {state: i for i, state in enumerate(unique_states)}

        # Initialize the transition matrix
        num_states = len(unique_states)
        P = np.zeros((num_states, num_states))

        # Populate the transition matrix based on state transitions
        for state, next_state in zip(states, next_states):
            P[state_to_index[state], state_to_index[next_state]] += 1

        # Normalize each row to obtain the transition probabilities
        P = P / P.sum(axis=1, keepdims=True)

        # Check for irreducibility and aperiodicity
        is_irreducible = self.verify_irreducibility(P)
        is_aperiodic = self.verify_aperiodicity(P)
        is_ergodic = is_irreducible and is_aperiodic

        # Log the results
        stability_log_path = os.path.join(self.save_path, f'markov_chain_stability_{run_name}.txt')
        with open(stability_log_path, 'w') as f:
            f.write(f"Markov Chain Stability Analysis:\n")
            f.write(f"Irreducible: {'Yes' if is_irreducible else 'No'}\n")
            f.write(f"Aperiodic: {'Yes' if is_aperiodic else 'No'}\n")
            f.write(f"Ergodic: {'Yes' if is_ergodic else 'No'}\n")

        return P, is_ergodic

    def verify_aperiodicity(self,P):
        """
        Check if the transition matrix P is aperiodic.
        Aperiodicity means that the greatest common divisor of the lengths of all cycles is 1.

        Args:
            P (np.array): Transition matrix of the Markov chain.

        Returns:
            bool: True if the chain is aperiodic, False otherwise.
        """
        # Check if all diagonal elements are greater than zero
        return np.all(np.diag(P) > 0)

    def verify_irreducibility(self, P):
        """
        Check if the transition matrix P is irreducible.
        Irreducibility means that every state can be reached from every other state.

        Args:
            P (np.array): Transition matrix of the Markov chain.

        Returns:
            bool: True if the chain is irreducible, False otherwise.
        """
        num_states = P.shape[0]
        # Raise P to the power of the number of states
        P_power = np.linalg.matrix_power(P, num_states)

        # Check if all elements are positive
        return np.all(P_power > 0)

    def verify_ergodicity(self, P):
        """Check if the Markov Chain is ergodic (irreducible and aperiodic)."""
        num_states = P.shape[0]
        # Check for irreducibility: all states should communicate
        irreducible = np.all(np.linalg.matrix_power(P, num_states) > 0)
        # Check for aperiodicity: the greatest common divisor of the cycle lengths is 1
        aperiodic = np.all(np.diag(P) > 0)
        return irreducible and aperiodic




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

        # Handle rows with zero sum to prevent division by zero
        for i in range(num_states):
            if P[i].sum() == 0:
                P[i] = np.ones(num_states) / num_states  # Assign uniform probability to all transitions

        # Normalize rows to get probabilities
        P = P / (P.sum(axis=1, keepdims=True) + 1e-10)  # Adding a small epsilon to avoid division by zero

        # Replace any NaNs or Infs in the transition matrix with zeros
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        # print(f"Normalized Transition Matrix:\n{P}")

        # Solve for the stationary distribution
        eigvals, eigvecs = np.linalg.eig(P.T)
        # print(f"Eigenvalues of Transition Matrix:\n{eigvals}")

        # Identify the eigenvectors corresponding to an eigenvalue of 1
        close_to_one = np.isclose(eigvals, 1)

        if not np.any(close_to_one):
            raise ValueError("No eigenvalues close to 1 found; unable to compute stationary distribution.")

        stationary_distribution = eigvecs[:, close_to_one]

        if stationary_distribution.size == 0:
            raise ValueError(
                "Stationary distribution is empty. This might indicate a problem with the transition matrix.")

        stationary_distribution = stationary_distribution[:, 0]
        stationary_distribution = stationary_distribution / stationary_distribution.sum()

        return unique_states, stationary_distribution

    def determine_stochastic_stability(self, stationary_distribution, unique_states):
        """Determine the stochastic stability of DFE, EE, or both (bistability)."""
        dfe_states = [state for state in unique_states if state[0] == 0]  # DFE where infected = 0
        ee_states = [state for state in unique_states if state[0] > 0]  # EE where infected > 0

        mu_dfe = sum(stationary_distribution[[unique_states.index(state) for state in dfe_states]])
        mu_ee = sum(stationary_distribution[[unique_states.index(state) for state in ee_states]])

        dfe_stability = mu_dfe > mu_ee
        ee_stability = mu_ee > mu_dfe
        bistability = mu_dfe == mu_ee

        return dfe_stability, ee_stability, bistability

    def plot_equilibrium_points_with_stationary_distribution(self, stationary_distribution, unique_states, run_name):
        infected = np.array([state[0] for state in unique_states])
        risk = np.array([state[1] for state in unique_states])

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(risk, infected, c=stationary_distribution, cmap='viridis', s=50, alpha=0.6)
        plt.colorbar(scatter, label='Stationary Distribution Probability')
        plt.xlabel('Community Risk')
        plt.ylabel('Number of Infected')
        plt.title(f'Equilibrium Points with Stationary Distribution (Run: {run_name})')

        # Highlight DFE and EE regions
        plt.axhline(y=0.5, color='r', linestyle='--', label='DFE/EE Boundary')
        plt.text(0.5, 0.25, 'DFE Region', horizontalalignment='center', color='red')
        plt.text(0.5, 0.75, 'EE Region', horizontalalignment='center', color='red')

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'equilibrium_points_stationary_{run_name}.png'))
        plt.close()

    def plot_transition_matrix_using_risk(self, states, next_states, community_risks, run_name,
                                          evaluation_subdirectory):
        """Plot and save the transition probability matrix using the extracted community risk values."""

        # Convert numpy arrays to tuples to make them hashable
        states = [tuple(state) for state in states]
        next_states = [tuple(next_state) for next_state in next_states]

        # Get unique states and map them to indices
        unique_states = sorted(set(states + next_states))
        state_to_idx = {state: idx for idx, state in enumerate(unique_states)}

        num_states = len(unique_states)

        # Initialize the transition matrix
        P = np.zeros((num_states, num_states))

        # Fill the transition matrix with community risk considerations
        for state, next_state, risk in zip(states, next_states, community_risks):
            transition_probability = risk  # Use community risk to weight transitions
            P[state_to_idx[state], state_to_idx[next_state]] += transition_probability

        # Normalize each row to get probabilities
        P = P / P.sum(axis=1, keepdims=True)

        # Avoid log(0) by setting a small floor value for the log scale
        P_log = np.log10(P + 1e-10)  # Adding a small epsilon to avoid log(0)

        # Plot the transition matrix with log scaling
        plt.figure(figsize=(10, 8))
        plt.imshow(P_log, cmap='gray', aspect='auto', origin='lower')
        plt.colorbar(label='Log10 of Transition Probability')
        plt.xlabel('Next State Index')
        plt.ylabel('Current State Index')
        plt.title(f'Transition Probability Matrix (Log Scale)\nRun: {run_name}')
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(evaluation_subdirectory, f'transition_matrix_{run_name}.png'))
        plt.close()

    def find_optimal_xy(self, infected_values, allowed_values, community_risk_values, z=95, run_name=None,
                        evaluation_subdirectory=None, alpha=0.5):
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

        # Calculate the final safety percentages
        time_above_optimal_x = sum(1 for val in infected_values if val > optimal_x)
        final_infection_safety_percentage = (time_above_optimal_x / len(infected_values)) * 100

        time_with_optimal_y_present = sum(1 for val in allowed_values if val >= optimal_y)
        final_attendance_safety_percentage = (time_with_optimal_y_present / len(allowed_values)) * 100

        # Determine if the safety conditions are met
        infection_condition_met = final_infection_safety_percentage <= (100 - z)
        attendance_condition_met = final_attendance_safety_percentage >= z

        # Save the safety condition results
        if run_name and evaluation_subdirectory:
            output_file_path = os.path.join(evaluation_subdirectory, f'safety_conditions_{run_name}_{alpha}.csv')

            with open(output_file_path, mode='w', newline='') as safety_file:
                safety_writer = csv.writer(safety_file)
                # Write the header
                safety_writer.writerow(['Run Name', 'Optimal X', 'Infection Safety %', 'Infection Condition Met',
                                        'Optimal Y', 'Attendance Safety %', 'Attendance Condition Met'])

                # Write the results
                safety_writer.writerow([
                    run_name,
                    optimal_x,
                    final_infection_safety_percentage,
                    'Yes' if infection_condition_met else 'No',
                    optimal_y,
                    final_attendance_safety_percentage,
                    'Yes' if attendance_condition_met else 'No'
                ])
                print(f"Saved safety conditions to {output_file_path}")

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
                state = torch.tensor([infected, risk], dtype=torch.float32)

                with torch.no_grad():
                    # q_values = self.model(state.unsqueeze(0))
                    # action = q_values.argmax(dim=1).item()
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    policy_logits, _ = self.model(state_tensor)
                    policy_dist = F.softmax(policy_logits, dim=-1)
                    action_index = torch.multinomial(policy_dist, 1).item()
                    scaled_action = self.scale_action(action_index, self.output_dim)
                    allowed_value = scaled_action * 50  # Convert to allowed values (0, 50, 100)

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

        # Calculate the 2D histogram with x as infected individuals and y as community risk
        hist, xedges, yedges = np.histogram2d(final_states[:, 0], final_states[:, 1], bins=50,
                                              range=[[0, 100], [0, 1]])

        # Plot the 2D histogram
        plt.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='gray_r')

        plt.colorbar(label='Frequency')
        plt.xlabel('Infected Individuals')
        plt.ylabel('Community Risk')
        plt.title(f'Simulated Steady-State Distribution\nRun: {run_name}, Alpha: {alpha}')

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'simulated_steady_state_{run_name}_alpha_{alpha}.png'))
        plt.close()

    def evaluate(self, run_name, num_episodes=1, x_value=38, y_value=80, z=95, alpha=0.5, csv_path=None):
        model_subdirectory = os.path.join(self.model_directory, self.agent_type, run_name)
        model_file_path = os.path.join(model_subdirectory, 'model.pt')

        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found in {model_file_path}")

        # Load the saved model
        self.load_model(run_name)

        # Set model to evaluation mode
        self.model.eval()
        print(f"Loaded model from {model_file_path}")

        if csv_path:
            self.community_risk_values = self.read_community_risk_from_csv(csv_path)
            self.max_weeks = len(self.community_risk_values)

        total_rewards = []
        evaluation_subdirectory = os.path.join(self.results_directory, run_name)
        os.makedirs(evaluation_subdirectory, exist_ok=True)
        csv_file_path = os.path.join(evaluation_subdirectory, f'evaluation_metrics_{run_name}.csv')

        allowed_values_over_time = []
        infected_values_over_time = []

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Step', 'State', 'Action', 'Infected', 'Allowed', 'Community Risk', 'Reward'])

            states = []
            next_states = []
            community_risks = []
            # allowed_values_over_time = []
            # infected_values_over_time = []

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
                        policy_logits, _ = self.model(state_tensor)
                        policy_dist = F.softmax(policy_logits, dim=-1)
                        action_index = torch.multinomial(policy_dist, 1).item()
                        scaled_action = self.scale_action(action_index, self.output_dim)

                    original_action = self.reverse_scale_action(scaled_action, self.output_dim)
                    next_state, reward, done, _, info = self.env.step((scaled_action, alpha))

                    next_state = np.array(next_state, dtype=np.float32)
                    total_reward += reward

                    # Try to extract infected value from both state and info
                    infected_value_state = next_state[0] if len(next_state) > 0 else None
                    infected_value_info = info.get('infected', None)

                    infected_value = infected_value_state if infected_value_state is not None else infected_value_info
                    allowed_value = scaled_action

                    states.append(state)
                    next_states.append(next_state)
                    community_risks.append(info['community_risk'])
                    infected_values_over_time.append(infected_value)
                    allowed_values_over_time.append(allowed_value)

                    writer.writerow([
                        episode + 1, step + 1, state.tolist(), scaled_action,
                        infected_value, allowed_value, info['community_risk'], reward
                    ])
                    state = next_state
                    step += 1

                total_rewards.append(total_reward)
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")

                # Calculate the percentage of time infections exceed the threshold
                time_exceeding_x = sum(1 for val in infected_values_over_time if val > x_value)
                time_within_x = len(infected_values_over_time) - time_exceeding_x
                infection_safety_percentage = (time_within_x / len(infected_values_over_time)) * 100

                time_with_y_present = sum(1 for val in allowed_values_over_time if val >= y_value)
                attendance_safety_percentage = (time_with_y_present / len(allowed_values_over_time)) * 100

                # Determine if the safety conditions are met
                infection_condition_met = infection_safety_percentage >= (
                        100 - z)  # At least 95% of time within safe infection range
                attendance_condition_met = attendance_safety_percentage >= z  # At least 95% of time with sufficient attendance

                infected_values, risk_values = self.simulate_states()
                lyapunov_model, loss_values = self.construct_lyapunov_function(infected_values, risk_values, alpha)

                # Plot Lyapunov function loss
                self.plot_loss_function(loss_values, alpha, run_name)

                # Plot steady-state distribution and stable points
                self.plot_steady_state_and_stable_points(lyapunov_model, run_name, alpha, infected_values, risk_values)

                # Plot Lyapunov properties
                mean_v, mean_delta_v = self.plot_lyapunov_properties(lyapunov_model, infected_values, risk_values,
                                                                     run_name, alpha)

                print(f"Mean Lyapunov Value: {mean_v}")
                print(f"Mean Change in Lyapunov Value: {mean_delta_v}")

                # Evaluate Lyapunov stability
                positive_definite, decreasing = self.evaluate_lyapunov_stability(lyapunov_model, states, alpha)

                print(f"Lyapunov Positive Definite: {positive_definite:.2%}")
                print(f"Lyapunov Decreasing: {decreasing:.2%}")

                optimal_x, optimal_y = self.find_optimal_xy(infected_values_over_time, allowed_values_over_time,
                                                            self.community_risk_values, z, run_name,
                                                            evaluation_subdirectory, alpha)

                print(f"Optimal x: {optimal_x}, Optimal y: {optimal_y}")
                allowed_barrier, infected_barrier, risk_barrier = self.simulate_states_with_barrier_control\
                        (infected_values, risk_values, optimal_x, optimal_y, self.output_dim, alpha, len(infected_values_over_time))

                self.plot_barrier_simulation_results(allowed_barrier, infected_barrier, risk_barrier, run_name,
                                        alpha)


            # Save final results
            with open(os.path.join(evaluation_subdirectory, 'final_results.txt'), 'w') as f:
                f.write(f"Cumulative Reward over {num_episodes} episodes: {sum(total_rewards)}\n")
                f.write(f"Average Reward: {np.mean(total_rewards)}\n")
                f.write(
                    f"Safety Percentage for Infections > {x_value}: {100 - infection_safety_percentage:.2f}% -> {'Condition Met' if infection_condition_met else 'Condition Not Met'}\n")
                f.write(
                    f"Safety Percentage for Attendance ≥ {y_value}: {attendance_safety_percentage:.2f}% -> {'Condition Met' if attendance_condition_met else 'Condition Not Met'}\n")


            min_length = min(len(allowed_values_over_time), len(infected_values_over_time),
                             len(self.community_risk_values))
            allowed_values_over_time = allowed_values_over_time[:min_length]
            infected_values_over_time = [20] + infected_values_over_time[:min_length - 1]
            community_risk_values = self.community_risk_values[:min_length]

            # Plotting
            x = np.arange(len(infected_values_over_time))
            fig, ax1 = plt.subplots(figsize=(12, 8))
            sns.set(style="whitegrid")

            # Bar plot for infected and allowed
            ax1.bar(x - 0.2, infected_values_over_time, width=0.4, color='red', label='Infected', alpha=0.6)
            ax1.bar(x + 0.2, allowed_values_over_time, width=0.4, color='blue', label='Allowed', alpha=0.6)

            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Infected / Allowed Values')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # Create a secondary y-axis for community risk
            ax2 = ax1.twinx()
            sns.lineplot(x=x, y=community_risk_values, marker='s', linestyle='--', color='black', linewidth=2.5, ax=ax2)
            ax2.set_ylabel('Community Risk')
            ax2.legend(loc='upper right')

            plt.title(f'Evaluation Results\nRun: {run_name}')

            # Save the plot
            plot_filename = os.path.join(self.save_path, f'evaluation_plot_{run_name}.png')
            plt.savefig(plot_filename)
            plt.close()

            self.log_all_states_visualizations(self.model, self.run_name, self.max_episodes, alpha,
                                               self.results_subdirectory)
            with open(os.path.join(evaluation_subdirectory, f"total_reward.txt"),
                      'a') as f:
                f.write(f"Total Reward: {sum(total_rewards)}\n")

        return total_rewards, allowed_values_over_time, infected_values_over_time, community_risk_values

    def apply_barrier_control(self, infected_value, risk_value, optimal_x, optimal_y, alpha, num_actions):
        """
        Apply barrier control to ensure the system stays within safe bounds using discrete allowed values.

        Args:
        infected_value (float): Current number of infected individuals.
        risk_value (float): Current community risk.
        optimal_x (float): Optimal infection threshold.
        optimal_y (float): Optimal attendance threshold.
        alpha (float): Alpha parameter for the policy.
        num_actions (int): The number of possible actions (discretized allowed values).

        Returns:
        int: Selected action index based on barrier control.
        """

        # Scale the infection and attendance thresholds (using reverse scaling logic)
        scaled_optimal_y = self.reverse_scale_action(optimal_y, num_actions)

        # If infection exceeds the threshold, reduce the allowed value (choose the lowest action)
        if infected_value > optimal_x:
            action_index = 0  # Corresponds to lowest allowed value (0)
        else:
            # If attendance is below threshold, increase the allowed value (choose the highest action)
            action_index = num_actions - 1 if risk_value < scaled_optimal_y else 0

        return action_index

    def simulate_states_with_barrier_control(self, infected_values, risk_values, optimal_x, optimal_y, num_actions, alpha=0.5,
                                             num_steps=100):
        """
        Simulate the system with barrier control applied at each step.

        Args:
        infected_values (numpy.ndarray): Array of initial infected values from simulate_states().
        risk_values (numpy.ndarray): Array of community risk values from simulate_states().
        optimal_x (float): Safe threshold for infections.
        optimal_y (float): Safe threshold for attendance.
        alpha (float): The alpha parameter for the policy.
        num_steps (int): Number of simulation steps.

        Returns:
        tuple: Lists of allowed values and updated infected values over time.
        """
        allowed_values_over_time = []
        infected_values_over_time = []
        risk_values_over_time = []

        for step in range(num_steps):
            # Get the current infected and risk values
            infected_value = infected_values[step]
            risk_value = risk_values[step]
            risk_values_over_time.append(risk_value)

            # Apply barrier control to determine the allowed student population
            allowed_value = self.scale_action(self.apply_barrier_control(infected_value, risk_value, optimal_x,
                                                                         optimal_y, alpha, num_actions), num_actions)
            allowed_values_over_time.append(allowed_value)

            # Simulate the new infected count based on the control action and dynamics
            alpha_infection = 0.005
            beta_infection = 0.01

            # Calculate new infection value based on current infected count and allowed students
            new_infected = ((alpha_infection * infected_value) * allowed_value) + \
                           ((beta_infection * risk_value) * allowed_value ** 2)

            # Update infected value for the next step
            infected_value = min(new_infected, allowed_value)
            infected_values_over_time.append(infected_value)

            print(f"Step {step + 1}: Allowed = {allowed_value}, Infected = {infected_value}, Risk = {risk_value}")

        return allowed_values_over_time, infected_values_over_time, risk_values_over_time



    def plot_barrier_simulation_results(self, allowed_values_over_time, infected_values_over_time, risk_values_over_time,
                                run_name, alpha):
        """
        Plot the simulation results with allowed values and infected values over time, and community risk as a subplot.

        Args:
        allowed_values_over_time (list): List of allowed student populations over time.
        infected_values_over_time (list): List of infected values over time.
        risk_values_over_time (list): List of community risk values over time.
        run_name (str): The name of the run for labeling the plot.
        alpha (float): The alpha parameter for labeling the plot.
        """
        time_steps = np.arange(len(allowed_values_over_time))

        # Create figure with 2 subplots, increase figure size
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

        # Increase bar width for better visibility and plot allowed and infected values
        bar_width = 0.4
        sns.barplot(x=time_steps, y=allowed_values_over_time, color='blue', label='Allowed Students', ax=ax1, alpha=0.7,
                    width=bar_width)
        sns.barplot(x=time_steps, y=infected_values_over_time, color='red', label='Infected Individuals', ax=ax1,
                    alpha=0.5, width=bar_width)

        # Set labels and legend for the first subplot
        ax1.set_ylabel('Count')
        ax1.legend(loc='upper left')
        ax1.set_title(f'A2C Barrier Control Simulation Results')

        # Plot community risk as a line plot in the second subplot
        sns.lineplot(x=time_steps, y=risk_values_over_time, color='green', marker='o', ax=ax2, label='Community Risk')
        ax2.set_ylabel('Community Risk')
        ax2.set_xlabel('Time Steps')
        ax2.legend(loc='upper left')

        # Rotate x-axis labels for visibility
        plt.xticks(rotation=45)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(self.save_path, f'barrier_simulation_results_{run_name}_alpha_{alpha}.png')
        plt.savefig(output_path)
        plt.show()

    def evaluate_lyapunov_stability(self, lyapunov_model, states, alpha):
        states_tensor = torch.tensor(states, dtype=torch.float32)

        with torch.no_grad():
            V = lyapunov_model(states_tensor).squeeze()
            next_states = self.get_next_states(states_tensor, alpha)
            V_next = lyapunov_model(next_states).squeeze()

            # Check positive definiteness
            positive_definite = (V > 0).float().mean()

            # Check if V is decreasing
            decreasing = (V_next < V).float().mean()

        return positive_definite.item(), decreasing.item()

    def simulate_states(self, num_simulations=100, num_steps=100, alpha=0.5):
        """
        Simulate the system for a long time to estimate the steady-state distribution.

        Args:
        num_simulations (int): Number of independent simulations to run.
        num_steps (int): Number of steps for each simulation.
        alpha (float): The alpha parameter for the policy.

        Returns:
        numpy.ndarray: Array of final states from all simulations.
        """
        infected_values = []
        community_risk_values = []

        for _ in range(num_simulations):
            infected = np.random.uniform(0, 100)  # Initial infected count
            risk = np.random.uniform(0, 1)  # Initial community risk

            for _ in range(num_steps):
                infected_values.append(infected)
                community_risk_values.append(risk)
                state = torch.tensor([infected, risk], dtype=torch.float32)

                with torch.no_grad():
                    # Get the policy_logits and value from the Actor-Critic model
                    policy_logits, _ = self.model(state.unsqueeze(0))

                    # Apply softmax to get action probabilities
                    policy_dist = F.softmax(policy_logits, dim=-1)

                    # Sample an action from the action distribution
                    action = torch.multinomial(policy_dist, 1).item()

                    allowed_value = self.scale_action(action, self.output_dim)


                alpha_infection = 0.005
                beta_infection = 0.01

                new_infected = ((alpha_infection * infected) * allowed_value) + \
                               ((beta_infection * risk) * allowed_value ** 2)

                infected = min(new_infected, allowed_value)
                risk = np.random.uniform(0, 1)  # Assuming risk is randomly distributed each step

        infected_values_array = np.array(infected_values)
        community_risk_values_array = np.array(community_risk_values)

        return infected_values_array, community_risk_values_array


def load_saved_model(model_directory, agent_type, run_name, input_dim, hidden_dim, num_actions):
    """Load a saved model's state dict from the subdirectory."""
    model_subdirectory = os.path.join(model_directory, agent_type, run_name)
    model_file_path = os.path.join(model_subdirectory, 'model.pt')

    if not os.path.exists(model_file_path):
        print(f"Model file not found at {model_file_path}")
        return None

    # Initialize a new model instance
    model = ActorCriticNetwork(input_dim, hidden_dim, num_actions)

    # Load the saved state dict
    model.load_state_dict(torch.load(model_file_path))
    model.eval()

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
