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
    file_paths = visualize_all_states(q_table, all_states, states, run_name, max_episodes, alpha, results_subdirectory)

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


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(ActorNetwork, self).__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_units):
            layer = nn.Linear(prev_dim, hidden_dim)
            setattr(self, f'fc{i + 1}', layer)  # This allows access to fc1, fc2, etc.
            layers.append(layer)
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)
        setattr(self, f'fc{len(hidden_units) + 1}', self.output_layer)  # This will be fc3 in the original setup

        self.network = nn.Sequential(*layers, self.output_layer)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = self.network(state)
        action_probs = F.softmax(x, dim=-1)
        return action_probs


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=1):
        super(CriticNetwork, self).__init__()
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_units):
            layer = nn.Linear(prev_dim, hidden_dim)
            setattr(self, f'fc{i + 1}', layer)  # This allows access to fc1, fc2, etc.
            layers.append(layer)
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)
        setattr(self, f'fc{len(hidden_units) + 1}', self.output_layer)  # This will be fc3 in the original setup

        self.network = nn.Sequential(*layers, self.output_layer)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        state_value = self.network(state)
        return state_value

class PPONetwork(nn.Module):
    def __init__(self, input_dim, actor_hidden_units, critic_hidden_units, output_dim):
        super(PPONetwork, self).__init__()
        self.actor = ActorNetwork(input_dim, actor_hidden_units, output_dim)
        self.critic = CriticNetwork(input_dim, critic_hidden_units)

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


class PPOCustomAgent:
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
        self.agent_type = "ppo_custom"
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

        # Replace DQN model with Actor and Critic networks
        self.actor = ActorNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.critic = CriticNetwork(self.input_dim, self.hidden_dim)


        # Initialize agent-specific configurations and variables

        # Initialize Actor-Critic specific parameters
        self.max_episodes = self.agent_config['agent']['max_episodes']
        self.discount_factor = self.agent_config['agent']['discount_factor']
        self.exploration_rate = self.agent_config['agent']['exploration_rate']
        self.min_exploration_rate = self.agent_config['agent']['min_exploration_rate']
        self.exploration_decay_rate = self.agent_config['agent']['exploration_decay_rate']

        # New Actor-Critic specific parameters
        self.actor_learning_rate = self.agent_config['agent']['actor_learning_rate']
        self.critic_learning_rate = self.agent_config['agent']['critic_learning_rate']
        self.actor_learning_rate_decay = self.agent_config['agent']['actor_learning_rate_decay']
        self.critic_learning_rate_decay = self.agent_config['agent']['critic_learning_rate_decay']
        self.min_actor_learning_rate = self.agent_config['agent']['min_actor_learning_rate']
        self.min_critic_learning_rate = self.agent_config['agent']['min_critic_learning_rate']
        self.entropy_coefficient = self.agent_config['agent']['entropy_coefficient']
        self.value_coefficient = self.agent_config['agent']['value_coefficient']
        self.max_grad_norm = self.agent_config['agent']['max_grad_norm']
        self.gae_lambda = self.agent_config['agent']['gae_lambda']
        self.clip_range = self.agent_config['agent']['clip_range']

        # Initialize Actor and Critic networks
        self.actor = ActorNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.critic = CriticNetwork(self.input_dim, self.hidden_dim)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        # Initialize learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1,
                                                         gamma=self.actor_learning_rate_decay)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1,
                                                          gamma=self.critic_learning_rate_decay)

        # Retain other relevant initializations
        self.decay_handler = ExplorationRateDecay(self.max_episodes, self.min_exploration_rate, self.exploration_rate)
        self.decay_function = self.agent_config['agent']['e_decay_function']
        self.softmax_temperature = self.agent_config['agent']['softmax_temperature']


        self.possible_actions = [list(range(0, (k))) for k in self.env.action_space.nvec]
        self.all_actions = [str(i) for i in list(itertools.product(*self.possible_actions))]

        # moving average for early stopping criteria
        self.moving_average_window = 100  # Number of episodes to consider for moving average
        self.stopping_criterion = 0.01  # Threshold for stopping
        self.prev_moving_avg = -float(
            'inf')  # Initialize to negative infinity to ensure any reward is considered an improvement in the first episode.

        # PPO-specific parameters
        self.clip_range = self.agent_config['agent']['clip_range']
        self.n_epochs = self.agent_config['agent']['n_epochs']
        self.batch_size = self.agent_config['agent']['batch_size']
        self.update_interval = self.agent_config['agent']['update_interval']
        self.mini_batch_size = self.agent_config['agent']['mini_batch_size']
        self.gae_lambda = self.agent_config['agent']['gae_lambda']

        # Early stopping
        self.early_stopping_patience = self.agent_config['agent']['early_stopping_patience']
        self.early_stopping_threshold = self.agent_config['agent']['early_stopping_threshold']

        # Logging and evaluation
        self.log_interval = self.agent_config['agent']['log_interval']
        self.eval_interval = self.agent_config['agent']['eval_interval']
        self.num_eval_episodes = self.agent_config['agent']['num_eval_episodes']

        # Network architecture
        self.actor_hidden_units = self.agent_config['agent']['actor_hidden_units']
        self.critic_hidden_units = self.agent_config['agent']['critic_hidden_units']

        if isinstance(self.actor_hidden_units, int):
            self.actor_hidden_units = [self.actor_hidden_units]
        if isinstance(self.critic_hidden_units, int):
            self.critic_hidden_units = [self.critic_hidden_units]

        # Learning rate for PPO network
        self.learning_rate = self.agent_config['agent']['learning_rate']

        # Update network initializations if using separate architectures
        self.actor = ActorNetwork(self.input_dim, self.actor_hidden_units, self.output_dim)
        self.critic = CriticNetwork(self.input_dim, self.critic_hidden_units, 1)

        self.ppo_network = PPONetwork(self.input_dim, self.actor_hidden_units, self.critic_hidden_units, self.output_dim)
        self.optimizer = optim.Adam(self.ppo_network.parameters(), lr=self.agent_config['agent']['learning_rate'])

        # 3. Ensure consistent use of learning rate decay
        self.learning_rate_decay = self.agent_config['agent']['learning_rate_decay']
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.learning_rate_decay)

        # 4. Add initialization for reward tracking (if not present elsewhere)
        self.episode_rewards = []
        self.best_average_reward = -float('inf')

        # 5. Initialize parameters for PPO update tracking
        self.steps_since_update = 0
        self.episodes_since_update = 0

        # 6. Initialize storage for collected experiences
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        # 7. Ensure the hidden_dim is consistent with actor_hidden_units
        self.hidden_dim = self.actor_hidden_units[-1]  # Assuming the last layer size is used

        # Update optimizer initialization
        self.optimizer = optim.Adam(self.ppo_network.parameters(), lr=self.agent_config['agent']['learning_rate'])

        # Hidden State
        self.hidden_state = None
        self.reward_window = deque(maxlen=self.moving_average_window)

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
        if random.random() < self.exploration_rate:
            return [self.scale_action(random.randint(0, self.output_dim - 1), self.output_dim) for _ in range(self.num_courses)]
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                action_probs, _ = self.ppo_network(state)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                return [self.scale_action(action.item(), self.output_dim)]



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

        Parameters:
        action_index (int): The index of the action.
        num_actions (int): The number of discrete actions available.

        Returns:
        int: The scaled action value.
        """
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than 1 to scale actions.")

        max_value = 100
        step_size = max_value / (num_actions - 1)
        return int(round(action_index * step_size))

    def reverse_scale_action(self, action, num_actions):
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than 1 to reverse scale actions.")
        max_value = 100
        step_size = max_value / (num_actions - 1)
        return round(action / step_size)

    def train(self, alpha):
        start_time = time.time()
        pbar = tqdm(total=self.max_episodes, desc="Training Progress", leave=True)

        actual_rewards = []
        predicted_rewards = []
        visited_state_counts = {}
        explained_variance_per_episode = []

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

        previous_values = None

        for episode in range(self.max_episodes):
            self.decay_handler.set_decay_function(self.decay_function)
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            done = False
            episode_rewards = []
            visited_states = set()
            episode_values = []
            step = 0
            value_change = 0
            episode_allowed = []
            episode_infected = []

            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action_probs, value = self.ppo_network(state_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                scaled_action = self.scale_action(action.item(), self.output_dim)
                next_state, reward, done, _, info = self.env.step(([scaled_action], alpha))
                next_state = np.array(next_state, dtype=np.float32)

                episode_allowed.append(sum(info.get('allowed', [])))
                episode_infected.append(sum(info.get('infected', [])))

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                log_probs.append(log_prob)
                values.append(value)

                total_reward += reward
                episode_rewards.append(reward)

                if previous_values is not None:
                    value_change += np.mean((value.detach().numpy() - previous_values) ** 2)
                previous_values = value.detach().numpy()

                episode_values.extend(value.detach().numpy().tolist())

                state = next_state
                state_tuple = tuple(state)
                visited_states.add(state_tuple)
                visited_state_counts[state_tuple] = visited_state_counts.get(state_tuple, 0) + 1
                step += 1

            # Compute advantages and returns
            advantages = self.compute_gae(rewards, values, dones)
            returns = [adv + value for adv, value in zip(advantages, values)]

            # PPO update
            self.ppo_update(states, actions, log_probs, returns, advantages)

            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)
            e_mean_allowed = sum(episode_allowed) / len(episode_allowed)
            e_mean_infected = sum(episode_infected) / len(episode_infected)
            allowed_means_per_episode.append(e_mean_allowed)
            infected_means_per_episode.append(e_mean_infected)
            actual_rewards.append(episode_rewards)
            predicted_rewards.append(episode_values)
            avg_episode_return = sum(episode_rewards) / len(episode_rewards)
            cumulative_reward = sum(episode_rewards)
            discounted_reward = sum([r * (self.discount_factor ** i) for i, r in enumerate(episode_rewards)])
            sample_efficiency = len(visited_states)

            metrics = {
                'episode': episode,
                'cumulative_reward': cumulative_reward,
                'average_reward': avg_episode_return,
                'discounted_reward': discounted_reward,
                'sample_efficiency': sample_efficiency,
                'policy_entropy': 0,  # You might want to compute this from action_probs
                'space_complexity': 0
            }
            writer.writerow(metrics)
            wandb.log({'cumulative_reward': cumulative_reward})

            pbar.update(1)
            pbar.set_description(f"Total Reward: {total_reward:.2f}, Epsilon: {self.exploration_rate:.2f}")

        pbar.close()
        csvfile.close()

        mean_allowed = round(sum(allowed_means_per_episode) / len(allowed_means_per_episode))
        mean_infected = round(sum(infected_means_per_episode) / len(infected_means_per_episode))

        summary_file_path = os.path.join(self.results_subdirectory, 'mean_allowed_infected.csv')
        with open(summary_file_path, 'w', newline='') as summary_csvfile:
            summary_writer = csv.DictWriter(summary_csvfile, fieldnames=['mean_allowed', 'mean_infected'])
            summary_writer.writeheader()
            summary_writer.writerow({'mean_allowed': mean_allowed, 'mean_infected': mean_infected})

        model = self.save_model()

        self.log_states_visited(list(visited_state_counts.keys()), list(visited_state_counts.values()), alpha,
                                self.results_subdirectory)

        return model

    def compute_gae(self, rewards, values, dones):
        gae = 0
        advantages = []
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[step + 1]

            delta = rewards[step] + self.discount_factor * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages

    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.stack(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        for _ in range(self.n_epochs):
            for batch_start in range(0, len(states), self.batch_size):
                batch_end = batch_start + self.batch_size
                state_batch = states[batch_start:batch_end]
                action_batch = actions[batch_start:batch_end]
                old_log_prob_batch = old_log_probs[batch_start:batch_end]
                return_batch = returns[batch_start:batch_end]
                advantage_batch = advantages[batch_start:batch_end]

                new_action_probs, new_values = self.ppo_network(state_batch)
                new_action_dist = torch.distributions.Categorical(new_action_probs)
                new_log_probs = new_action_dist.log_prob(action_batch)

                ratio = (new_log_probs - old_log_prob_batch).exp()
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values.squeeze(), return_batch)

                entropy = new_action_dist.entropy().mean()

                loss = actor_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ppo_network.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save_model(self):
        model_path = os.path.join(self.model_subdirectory, 'ppo_model.pt')
        torch.save(self.ppo_network.state_dict(), model_path)
        print(f"Model saved at: {model_path}")

    def load_model(self):
        model_path = os.path.join(self.model_subdirectory, 'ppo_model.pt')
        if os.path.exists(model_path):
            self.ppo_network.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Model loaded from: {model_path}")
        else:
            print("No saved model found. Starting with a new model.")

    def generate_all_states(self):
        value_range = range(0, 101, 10)
        input_dim = self.ppo_network.actor.input_dim

        if input_dim == 2:
            all_states = [np.array([i, j]) for i in value_range for j in value_range]
        else:
            course_combinations = itertools.product(value_range, repeat=self.num_courses)
            all_states = [np.array(list(combo) + [risk]) for combo in course_combinations for risk in value_range]
            all_states = [
                state[:input_dim] if len(state) > input_dim else
                np.pad(state, (0, max(0, input_dim - len(state))), 'constant')
                for state in all_states
            ]

        return all_states
    def log_all_states_visualizations(self, model, run_name, max_episodes, alpha, results_subdirectory):
        all_states = self.generate_all_states()
        num_courses = len(self.env.students_per_course)
        file_paths = visualize_all_states(model, all_states, run_name, num_courses, max_episodes, alpha,
                                          results_subdirectory)
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
        self.reward_window = deque(maxlen=self.moving_average_window)
        self.actor = ActorNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        self.critic = CriticNetwork(self.input_dim, self.hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=100, gamma=self.actor_learning_rate_decay)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=100, gamma=self.critic_learning_rate_decay)

        self.run_rewards_per_episode = []  # Store rewards per episode for this run

        pbar = tqdm(total=self.max_episodes, desc=f"Training Run {seed}", leave=True)
        visited_state_counts = {}
        previous_value = None

        for episode in range(self.max_episodes):
            self.decay_handler.set_decay_function(self.decay_function)
            state, _ = self.env.reset()
            state = np.array(state, dtype=np.float32)
            total_reward = 0
            done = False
            episode_rewards = []
            visited_states = set()  # Using a set to track unique states
            episode_values = []
            step = 0
            value_change = 0

            while not done:
                actions = self.select_action(state)
                next_state, reward, done, _, info = self.env.step((actions, alpha))
                next_state = np.array(next_state, dtype=np.float32)

                original_actions = [action // 50 for action in actions]
                total_reward += reward
                episode_rewards.append(reward)

                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                action_tensor = torch.LongTensor(original_actions)
                reward_tensor = torch.FloatTensor([reward])

                # Critic update
                current_value = self.critic(state_tensor)
                next_value = self.critic(next_state_tensor)
                target_value = reward_tensor + (1 - done) * self.discount_factor * next_value
                critic_loss = F.mse_loss(current_value, target_value.detach())

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Actor update
                action_probs = self.actor(state_tensor)
                action_distribution = torch.distributions.Categorical(action_probs)
                log_prob = action_distribution.log_prob(action_tensor)
                entropy = action_distribution.entropy().mean()
                advantage = (target_value - current_value).detach()
                actor_loss = -(log_prob * advantage).mean() - self.entropy_coefficient * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                if previous_value is not None:
                    value_change += np.mean((current_value.detach().numpy() - previous_value) ** 2)
                previous_value = current_value.detach().numpy()

                episode_values.extend(current_value.detach().numpy().tolist())

                state = next_state
                state_tuple = tuple(state)
                visited_states.add(state_tuple)
                visited_state_counts[state_tuple] = visited_state_counts.get(state_tuple, 0) + 1
                step += 1

            self.exploration_rate = self.decay_handler.get_exploration_rate(episode)
            self.run_rewards_per_episode.append(episode_rewards)
            self.actor_scheduler.step()
            self.critic_scheduler.step()
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

    def construct_lyapunov_function(self, features, alpha):
        model = LyapunovNet(input_dim=2, hidden_dim=64, output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_values = []
        epochs = 1000
        epsilon = 1e-6

        train_states = torch.tensor([[f[0], f[1]] for f in features], dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            V = model(train_states)
            next_states = self.get_next_states(train_states, alpha)
            V_next = model(next_states)

            positive_definite_loss = F.relu(-V + epsilon).mean()
            decreasing_loss = F.relu(V_next - V + epsilon).mean()

            zero_states = torch.all(train_states == 0, dim=1)
            origin_loss = V[zero_states].mean() if zero_states.any() else torch.tensor(0.0)

            loss = positive_definite_loss + decreasing_loss + origin_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf detected at epoch {epoch}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_values.append(loss.item())

        torch.save(model.state_dict(), os.path.join(self.save_path, 'lyapunov_model.pth'))
        with open(os.path.join(self.save_path, 'lyapunov_loss_values.txt'), 'w') as f:
            for i, loss_val in enumerate(loss_values):
                f.write(f"Epoch {i}: Loss = {loss_val}\n")

        return model, None, loss_values  # Return None for theta as it's not used in this implementation

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
                q_values = self.model(state.unsqueeze(0))
                action = q_values.argmax(dim=1).item()
                allowed_value = action * 50  # Convert to allowed values (0, 50, 100)

            current_infected = state[0].item()
            community_risk = state[1].item()

            alpha_infection = 0.005
            beta_infection = 0.01

            new_infected = ((alpha_infection * current_infected) * allowed_value) + \
                           ((beta_infection * community_risk) * allowed_value ** 2)

            new_infected = min(new_infected, allowed_value)
            next_states.append(torch.tensor([new_infected, community_risk], dtype=torch.float32))

        return torch.stack(next_states)

    def plot_steady_state_and_stable_points(self, V, features, run_name, alpha):
        infected = np.linspace(0, 100, 100)
        risk = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(infected, risk)
        states = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)

        with torch.no_grad():
            V_values = V(states).squeeze().cpu().numpy().reshape(X.shape)

        steady_state = np.exp(-V_values / V_values.max())
        steady_state /= steady_state.sum()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        c1 = ax1.contourf(X, Y, steady_state, levels=20, cmap='viridis')
        ax1.set_title(f'Steady-State Distribution (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('Infected')
        ax1.set_ylabel('Community Risk')
        fig.colorbar(c1, ax=ax1, label='Steady-State Probability')

        c2 = ax2.contourf(X, Y, V_values, levels=20, cmap='coolwarm')
        ax2.set_title(f'Lyapunov Function (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('Infected')
        ax2.set_ylabel('Community Risk')
        fig.colorbar(c2, ax=ax2, label='Lyapunov Function Value')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'steady_state_and_stable_points_{run_name}_alpha_{alpha}.png'))
        plt.close()

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

        im1 = ax1.imshow(V_values, extent=[0, 100, 0, 1], origin='lower', aspect='auto', cmap='viridis')
        ax1.set_title(f'Lyapunov Function V(x) (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('Infected')
        ax1.set_ylabel('Community Risk')
        fig.colorbar(im1, ax=ax1, label='V(x)')

        im2 = ax2.imshow(delta_V, extent=[0, 100, 0, 1], origin='lower', aspect='auto', cmap='coolwarm')
        ax2.set_title(f'Change in Lyapunov Function ΔV(x) (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('Infected')
        ax2.set_ylabel('Community Risk')
        fig.colorbar(im2, ax=ax2, label='ΔV(x)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'lyapunov_change_{run_name}_alpha_{alpha}.png'))
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
        ax1.plot(V_values.cpu().numpy(), label='V(x)')
        ax1.axhline(y=0, color='r', linestyle='--', label='V(x) = 0')
        ax1.set_title(f'Lyapunov Function Values (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('State Index')
        ax1.set_ylabel('V(x)')
        ax1.legend()
        ax1.grid(True)

        # Plot ΔV(x)
        ax2.plot(delta_V.cpu().numpy(), label='ΔV(x)')
        ax2.axhline(y=0, color='r', linestyle='--', label='ΔV(x) = 0')
        ax2.set_title(f'Change in Lyapunov Function (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('State Index')
        ax2.set_ylabel('ΔV(x)')
        ax2.legend()
        ax2.grid(True)

        # Plot V(x) vs ΔV(x)
        ax3.scatter(V_values.cpu().numpy(), delta_V.cpu().numpy(), alpha=0.6)
        ax3.axhline(y=0, color='r', linestyle='--', label='ΔV(x) = 0')
        ax3.axvline(x=0, color='g', linestyle='--', label='V(x) = 0')
        ax3.set_title(f'V(x) vs ΔV(x) (Run: {run_name}, Alpha: {alpha})')
        ax3.set_xlabel('V(x)')
        ax3.set_ylabel('ΔV(x)')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'lyapunov_properties_{run_name}_alpha_{alpha}.png'))
        plt.close()

        positive_definite = (V_values > 0).float().mean()
        decreasing = (delta_V < 0).float().mean()

        print(f"Positive definite: {positive_definite.item():.2%}")
        print(f"Decreasing: {decreasing.item():.2%}")

        return positive_definite.item(), decreasing.item()
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
                    q_values = self.model(state.unsqueeze(0))
                    action = q_values.argmax().item()
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
        plt.savefig(os.path.join(self.save_path, f'simulated_steady_state_{run_name}_alpha_{alpha}.png'))
        plt.close()



    def evaluate(self, run_name, num_episodes=1, x_value=38, y_value=80, z=95, alpha=0.5, csv_path=None):
        # print('Alpha from main:', alpha)
        model_subdirectory = os.path.join(self.model_directory, self.agent_type, run_name)
        self.model_subdirectory = model_subdirectory
        results_directory = self.results_subdirectory

        try:
            self.load_model()
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            return []

        self.actor.eval()
        self.critic.eval()
        print(f"Loaded models from {model_subdirectory}")

        if csv_path:
            self.community_risk_values = self.read_community_risk_from_csv(csv_path)
            self.max_weeks = len(self.community_risk_values)
            # print(f"Community Risk Values: {self.community_risk_values}")

        total_rewards = []
        evaluation_subdirectory = os.path.join(results_directory, run_name)
        os.makedirs(evaluation_subdirectory, exist_ok=True)
        csv_file_path = os.path.join(evaluation_subdirectory, f'evaluation_metrics_{run_name}.csv')

        safety_log_path = os.path.join(evaluation_subdirectory, f'safety_conditions_{run_name}.csv')
        interpretation_path = os.path.join(evaluation_subdirectory, f'safety_conditions_interpretation_{run_name}.txt')

        allowed_values_over_time = []
        infected_values_over_time = []

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Step', 'State', 'Action', 'Community Risk', 'Total Reward'])

            states = []
            next_states = []
            community_risks = []

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
                        action_probs = self.actor(state_tensor)
                        action_distribution = torch.distributions.Categorical(action_probs)
                        action = action_distribution.sample()
                        scaled_action = self.scale_action(action.item(), self.output_dim)

                    next_state, reward, done, _, info = self.env.step(([scaled_action], alpha))
                    next_state = np.array(next_state, dtype=np.float32)
                    total_reward += reward
                    states.append(state)
                    next_states.append(next_state)
                    community_risks.append(info['community_risk'])

                    writer.writerow(
                        [episode + 1, step + 1, int(state[0]), scaled_action, info['community_risk'], reward]
                    )

                    if episode == 0:
                        allowed_values_over_time.append(scaled_action)
                        infected_values_over_time.append(next_state[0])

                    state = next_state
                    step += 1

                total_rewards.append(total_reward)
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")

                # Ensure all arrays have the same length
                min_length = min(len(allowed_values_over_time), len(infected_values_over_time),
                                 len(self.community_risk_values))
                allowed_values_over_time = allowed_values_over_time[:min_length]
                infected_values_over_time = infected_values_over_time[:min_length]
                community_risk_values = self.community_risk_values[:min_length]

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

                # Construct CBFs using direct x and y values
                B1, B2 = self.construct_cbf(allowed_values_over_time, infected_values_over_time,
                                            evaluation_subdirectory,
                                            x_value, y_value)

                # Verify forward invariance
                is_invariant = self.verify_forward_invariance(B1, B2, allowed_values_over_time,
                                                              infected_values_over_time, evaluation_subdirectory)

                # After all episodes have been evaluated and the CSV file has been saved, call the transition matrix plotting function
                self.plot_transition_matrix_using_risk(states, next_states, community_risks, run_name,
                                                       evaluation_subdirectory)
                # Call find_optimal_xy to determine the best x and y and plot the safety conditions
                optimal_x, optimal_y = self.find_optimal_xy(infected_values_over_time, allowed_values_over_time,
                                                            self.community_risk_values, z, run_name,
                                                            evaluation_subdirectory, alpha)

                print(f"Optimal x: {optimal_x}, Optimal y: {optimal_y}")

            # Save final results
            with open(os.path.join(evaluation_subdirectory, 'final_results.txt'), 'w') as f:
                f.write(f"Cumulative Reward over {num_episodes} episodes: {sum(total_rewards)}\n")
                f.write(f"Average Reward: {np.mean(total_rewards)}\n")
                f.write(
                    f"Safety Percentage for Infections > {x_value}: {100 - infection_safety_percentage:.2f}% -> {'Condition Met' if infection_condition_met else 'Condition Not Met'}\n")
                f.write(
                    f"Safety Percentage for Attendance ≥ {y_value}: {attendance_safety_percentage:.2f}% -> {'Condition Met' if attendance_condition_met else 'Condition Not Met'}\n")
                f.write(f"Forward Invariance Verified: {'Yes' if is_invariant else 'No'}\n")

            min_length = min(len(allowed_values_over_time), len(infected_values_over_time),
                             len(self.community_risk_values))
            allowed_values_over_time = allowed_values_over_time[:min_length]
            infected_values_over_time = [20] + infected_values_over_time[:min_length-1]
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
            # ax2.plot(x, community_risk_values, color='black', marker='o', label='Community Risk')
            sns.lineplot(x=x, y=community_risk_values, marker='s', linestyle='--', color='black', linewidth=2.5, ax=ax2)
            ax2.set_ylabel('Community Risk')
            ax2.legend(loc='upper right')

            plt.title(f'Evaluation Results\nRun: {run_name}')

            # Save the plot
            plot_filename = os.path.join(self.save_path, f'evaluation_plot_{run_name}.png')
            plt.savefig(plot_filename)
            plt.close()
            self.load_model()  # This should load the combined PPO network

            # Generate all states
            all_states = self.generate_all_states()

            # Update the visualization function call
            self.log_all_states_visualizations(self.ppo_network.actor, self.run_name, self.max_episodes, alpha,
                                               self.results_subdirectory)

        return total_rewards


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
