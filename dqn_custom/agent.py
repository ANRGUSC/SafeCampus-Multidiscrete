import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import logging
from collections import deque
import random
import itertools
from tqdm import tqdm
from .utilities import load_config
from .visualizer import visualize_all_states, visualize_q_table, visualize_variance_in_rewards_heatmap, \
    visualize_explained_variance, visualize_variance_in_rewards, visualize_infected_vs_community_risk_table, \
    states_visited_viz
import wandb
from torch.optim.lr_scheduler import StepLR
import math
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import torch.nn.functional as F

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
    wandb_images = [wandb.Image(path) for path in file_paths]
    wandb.log({"All States Visualization": wandb_images})

    # Log them individually with dimension information
    for path in file_paths:
        infected_dim = path.split('infected_dim_')[-1].split('.')[0]
        wandb.log({f"All States Visualization (Infected Dim {infected_dim})": wandb.Image(path)})


def log_states_visited(states, visit_counts, alpha, results_subdirectory):
    file_paths = states_visited_viz(states, visit_counts, alpha, results_subdirectory)

    # Log all generated heatmaps
    wandb_images = [wandb.Image(path) for path in file_paths]
    wandb.log({"States Visited": wandb_images})

    # Log them individually with dimension information
    for path in file_paths:
        if "error" in path:
            wandb.log({"States Visited Error": wandb.Image(path)})
        else:
            dim = path.split('infected_dim_')[-1].split('.')[0]
            wandb.log({f"States Visited (Infected Dim {dim})": wandb.Image(path)})

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
            # exploration_rate = self.min_exploration_rate + (
            #             self.initial_exploration_rate - self.min_exploration_rate) / (
            #                                1 + np.exp(episode - self.max_episodes / 2))
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
    def __init__(self, env, run_name, shared_config_path, agent_config_path=None, override_config=None):
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
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.agent_type = "dqn_custom"
        self.run_name = run_name
        self.results_subdirectory = os.path.join(self.results_directory, self.agent_type, self.run_name, self.timestamp)
        if not os.path.exists(self.results_subdirectory):
            os.makedirs(self.results_subdirectory, exist_ok=True)
        self.model_directory = self.shared_config['directories']['model_directory']
        self.model_subdirectory = os.path.join(self.model_directory, self.agent_type, self.run_name, self.timestamp)
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
        self.replay_memory = deque(maxlen=self.agent_config['agent']['replay_memory_capacity'])
        self.batch_size = self.agent_config['agent']['batch_size']

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
    def train(self, alpha):
        pbar = tqdm(total=self.max_episodes, desc="Training Progress", leave=True)

        actual_rewards = []
        predicted_rewards = []
        visited_state_counts = {}
        explained_variance_per_episode = []

        for episode in range(self.max_episodes):
            self.decay_handler.set_decay_function(self.decay_function)
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            done = False
            episode_rewards = []
            visited_states = []
            episode_q_values = []

            while not done:
                actions = self.select_action(state)
                next_state, reward, done, _, info = self.env.step((actions, alpha))
                next_state = np.array(next_state, dtype=np.float32)

                # When storing in replay memory, store the original action indices
                original_actions = [action // 50 for action in actions]
                self.replay_memory.append((state, original_actions, reward, next_state, done))

                state = next_state
                total_reward += reward
                episode_rewards.append(reward)

                state_tuple = tuple(state)
                visited_states.append(state_tuple)
                visited_state_counts[state_tuple] = visited_state_counts.get(state_tuple, 0) + 1
                # print(info)

                if len(self.replay_memory) > self.batch_size:
                    batch = random.sample(self.replay_memory, self.batch_size)
                    states, actions, rewards_batch, next_states, dones = map(np.array, zip(*batch))

                    states = torch.FloatTensor(states)
                    actions = torch.LongTensor(actions)
                    rewards_batch = torch.FloatTensor(rewards_batch)
                    next_states = torch.FloatTensor(next_states)
                    dones = torch.FloatTensor(dones)

                    current_q_values = self.model(states)
                    # print(f"Current Q-values shape: {current_q_values.shape}")
                    # print(f"Actions shape: {actions.shape}")
                    # print(f"Sample of actions: {actions[:5]}")

                    # Handle multi-course scenario
                    batch_size, num_actions = current_q_values.shape
                    num_courses = actions.shape[1]

                    # Reshape current_q_values to [batch_size * num_courses, num_actions]
                    current_q_values = current_q_values.repeat(1, num_courses).view(-1, num_actions)

                    # Flatten actions to [batch_size * num_courses]
                    actions = actions.view(-1)

                    # Gather the Q-values for the taken actions
                    current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                    # Reshape back to [batch_size, num_courses]
                    current_q_values = current_q_values.view(batch_size, num_courses)

                    next_q_values = self.model(next_states)
                    next_q_values = next_q_values.repeat(1, num_courses).view(-1, num_actions).max(1)[0].view(
                        batch_size, num_courses)

                    # Sum Q-values across courses
                    current_q_values = current_q_values.sum(1)
                    next_q_values = next_q_values.sum(1)

                    target_q_values = rewards_batch + (1 - dones) * self.discount_factor * next_q_values

                    loss = nn.MSELoss()(current_q_values, target_q_values)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    episode_q_values.extend(current_q_values.detach().numpy().tolist())

            actual_rewards.append(episode_rewards)
            predicted_rewards.append(episode_q_values)

            if episode_q_values:
                explained_variance = self.calculate_explained_variance(episode_rewards, episode_q_values)
            else:
                explained_variance = 0  # or some default value
            explained_variance_per_episode.append(explained_variance)

            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)

            wandb.log({
                "total_reward": total_reward,
                "exploration_rate": self.exploration_rate,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "loss": loss.item() if 'loss' in locals() else 0,
                "avg_reward": np.mean(episode_rewards),
            })

            pbar.update(1)
            pbar.set_description(
                f"Total Reward: {total_reward:.2f}, Epsilon: {self.exploration_rate:.2f}")

        pbar.close()

        # After training, save the model
        model_file_path = os.path.join(self.model_subdirectory, 'model.pt')
        torch.save(self.model.state_dict(), model_file_path)

        # Visualization and logging
        saved_model = load_saved_model(self.model_directory, self.agent_type, self.run_name, self.timestamp,
                                       self.input_dim, self.hidden_dim, self.output_dim)
        value_range = range(0, 101, 10)
        all_states = self.generate_all_states()
        # all_states_path = visualize_all_states(saved_model, all_states, self.run_name, self.max_episodes, alpha,
        #                                        self.results_subdirectory)
        # wandb.log({"All_States_Visualization": [wandb.Image(all_states_path)]})
        states = list(visited_state_counts.keys())
        visit_counts = list(visited_state_counts.values())
        self.log_states_visited(states, visit_counts, alpha, self.results_subdirectory)

        self.log_all_states_visualizations(self.model, self.run_name, self.max_episodes, alpha, self.results_subdirectory)


        # states = list(visited_state_counts.keys())
        # visit_counts = list(visited_state_counts.values())
        # states_visited_path = states_visited_viz(states, visit_counts, alpha, self.results_subdirectory)
        # wandb.log({"States Visited": [wandb.Image(states_visited_path)]})

        avg_rewards = [sum(lst) / len(lst) for lst in actual_rewards]
        explained_variance_path = os.path.join(self.results_subdirectory, 'explained_variance.png')
        visualize_explained_variance(explained_variance_per_episode, explained_variance_path)
        wandb.log({"Explained Variance": [wandb.Image(explained_variance_path)]})

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

        # Log all generated visualizations
        # wandb_images = [wandb.Image(path) for path in file_paths]
        # wandb.log({"All States Visualization": wandb_images})

        # Log them individually
        # for path in file_paths:
        #     if "infected_vs_community_risk" in path:
        #         wandb.log({"All States Visualization (Infected vs Community Risk)": wandb.Image(path)})
        #     elif "vs_community_risk" in path:
        #         course = path.split('course_')[1].split('_vs')[0]
        #         wandb.log({f"All States Visualization (Course {course} vs Community Risk)": wandb.Image(path)})
        #     elif "vs_course" in path:
        #         courses = path.split('course_')[1].split('.')[0]
        #         wandb.log({f"All States Visualization (Course {courses})": wandb.Image(path)})
    def log_states_visited(self, states, visit_counts, alpha, results_subdirectory):
        file_paths = states_visited_viz(states, visit_counts, alpha, results_subdirectory)
        print("file_paths: ", file_paths)

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

        for episode in range(self.max_episodes):
            self.decay_handler.set_decay_function(self.decay_function)
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            done = False
            episode_rewards = []
            visited_states = []
            loss = torch.tensor(0.0)  # Initialize loss here
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step((action, alpha))
                next_state = np.array(next_state, dtype=np.float32)

                # When storing in replay memory, store the original action indices
                original_actions = [action // 50 for action in actions]
                self.replay_memory.append((state, original_actions, reward, next_state, done))
                state = next_state
                total_reward += reward
                episode_rewards.append(reward)

                state_tuple = tuple(state)
                visited_states.append(state_tuple)
                visited_state_counts[state_tuple] = visited_state_counts.get(state_tuple, 0) + 1

                if len(self.replay_memory) > self.batch_size:
                    batch = random.sample(self.replay_memory, self.batch_size)
                    states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

                    states = torch.FloatTensor(states)
                    actions = torch.LongTensor(actions)
                    rewards = torch.FloatTensor(rewards)
                    next_states = torch.FloatTensor(next_states)
                    dones = torch.FloatTensor(dones)

                    current_q_values = self.model(states)
                    current_q_values = current_q_values.view(self.batch_size, self.num_courses, -1)
                    current_q_values = current_q_values.gather(2, actions.unsqueeze(2)).squeeze(2)

                    next_q_values = self.model(next_states).view(self.batch_size, self.num_courses, -1).max(2)[0]
                    target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values.sum(dim=1)

                    loss = nn.MSELoss()(current_q_values.sum(dim=1), target_q_values)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            self.run_rewards_per_episode.append(episode_rewards)
            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)

            pbar.update(1)
            pbar.set_description(
                f"Loss:{loss}, Total Reward: {total_reward:.2f}, Epsilon: {self.exploration_rate:.2f}")

        pbar.close()


        return self.run_rewards_per_episode

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
        episodes = list(range(num_episodes))  # Assume all runs have the same number of episodes)

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

        # Check for NaNs and Infs
        if np.any(np.isnan(lower_bounds)) or np.any(np.isnan(upper_bounds)) or np.any(np.isnan(central_tendency)):
            raise ValueError("Array contains NaNs.")
        if np.any(np.isinf(lower_bounds)) or np.any(np.isinf(upper_bounds)) or np.any(np.isinf(central_tendency)):
            raise ValueError("Array contains Infs.")

        # Smoothing the curve
        spline_points = 300  # Number of points for spline interpolation
        episodes_smooth = np.linspace(episodes[0], episodes[-1], spline_points)
        central_tendency_smooth = make_interp_spline(episodes, central_tendency)(episodes_smooth)
        lower_bounds_smooth = make_interp_spline(episodes, lower_bounds)(episodes_smooth)
        upper_bounds_smooth = make_interp_spline(episodes, upper_bounds)(episodes_smooth)

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
        spline_points = 300  # Number of points for spline interpolation
        episodes_smooth = np.linspace(episodes[0], episodes[-1], spline_points)
        means_smooth = make_interp_spline(episodes, means)(episodes_smooth)
        lower_bounds_smooth = make_interp_spline(episodes, lower_bounds)(episodes_smooth)
        upper_bounds_smooth = make_interp_spline(episodes, upper_bounds)(episodes_smooth)

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
        boxplot_output_path = os.path.join(self.results_subdirectory, 'boxplot_confidence_interval.png')
        self.visualize_boxplot_confidence_interval(returns_per_episode, confidence_alpha, boxplot_output_path)
        wandb.log({"Box Plot Confidence Interval": [wandb.Image(boxplot_output_path)]})


def load_saved_model(model_directory, agent_type, run_name, timestamp, input_dim, hidden_dim, action_space_nvec):
    """
    Load a saved DeepQNetwork model from the subdirectory.

    Args:
    model_directory: Base directory where models are stored.
    agent_type: Type of the agent, used in directory naming.
    run_name: Name of the run, used in directory naming.
    timestamp: Timestamp of the model saving time, used in directory naming.
    input_dim: Input dimension of the model.
    hidden_dim: Hidden layer dimension.
    action_space_nvec: Action space vector size.

    Returns:
    model: The loaded DeepQNetwork model, or None if loading failed.
    """
    # Construct the model subdirectory path
    model_subdirectory = os.path.join(model_directory, agent_type, run_name, timestamp)

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
