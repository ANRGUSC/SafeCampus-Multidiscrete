import yaml
import logging

import os
import random
import numpy as np
import torch
import csv

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


        elif self.current_decay_function == 3:  # Polynomial Decay with Decay Rate

            exploration_rate = self.initial_exploration_rate * (1 - (episode / self.max_episodes) ** 3)


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

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
