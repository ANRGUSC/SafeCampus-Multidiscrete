import math
import copy
import random
import csv
import logging
from epidemic_models.analyze_models import estimate_infected_students
import numpy as np

random.seed(100)
np.random.seed(100)


class Simulation:
    def __init__(self, model, read_community_risk_from_csv=False, csv_path=None, mode='train'):
        self.current_time = 0
        self.model = model
        self.allowed_students_per_course = []
        self.student_status = [20] * model.num_courses
        self.state_transition = []
        self.weekly_infected_students = []
        self.allowed = []
        self.infected = []
        self.community_risk = random.random()  # Default random value
        self.max_weeks = model.get_max_weeks()
        self.total_steps = 0
        self.episode_count = 0
        self.episode_seed = 0
        self.risk_values = self.generate_episode_risk()
        self.risk_iterator = iter(self.risk_values)
        self.community_risk = next(self.risk_iterator)
        self.mode = mode  # 'train' or 'eval'

        if read_community_risk_from_csv and csv_path:
            self.read_community_risk_from_csv(csv_path)
        else:
            self.risk_generator = self.generate_episode_risk()
            self.risk_values = self.generate_episode_risk()
            self.risk_iterator = iter(self.risk_values)
            self.community_risk = next(self.risk_iterator)

        # Logging max_weeks from model
        logging.info(f"Simulation initialized with max weeks: {self.model.get_max_weeks()}")

    def generate_episode_risk(self):
        """Generate risk values for a single episode."""
        self.episode_seed += 1
        random.seed(self.episode_seed)
        np.random.seed(self.episode_seed)

        t = np.linspace(0, 2 * np.pi, self.max_weeks)
        num_components = random.randint(1, 3)  # Use 1 to 3 sine components
        risk_pattern = np.zeros(self.max_weeks)

        for _ in range(num_components):
            amplitude = random.uniform(0.2, 0.4)
            frequency = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2 * np.pi)
            risk_pattern += amplitude * np.sin(frequency * t + phase)

        risk_pattern = (risk_pattern - np.min(risk_pattern)) / (np.max(risk_pattern) - np.min(risk_pattern))
        risk_pattern = 0.9 * risk_pattern + 0.0  # Scale to range [0.1, 0.9]

        return [max(0.0, min(1.0, risk + random.uniform(-0.1, 0.1))) for risk in risk_pattern]

    def read_community_risk_from_csv(self, csv_path):
        try:
            with open(csv_path, mode='r') as file:
                reader = csv.DictReader(file)
                self.community_risk_values = [float(row['Risk-Level']) for row in reader]
                logging.info(f"Community risk values read from CSV: {self.community_risk_values}")
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            raise ValueError(f"Error reading CSV file: {e}")

    def set_community_risk_high(self):
        self.community_risk = random.uniform(0.055, 0.05)
        return self.community_risk

    def set_community_risk_low(self):
        self.community_risk = random.uniform(0.01, 0.025)
        return self.community_risk

    def get_student_status(self):
        obs_state = copy.deepcopy(self.student_status)
        obs_state.append(int(self.community_risk * 100))
        return obs_state

    def update_with_action(self, action):
        if self.current_time < self.model.get_max_weeks():
            self.apply_action(action, self.community_risk)
        return None

    def apply_action(self, action: list, community_risk: float):
        allowed_students_per_course = []
        for i, students in enumerate(self.model.number_of_students_per_course()):
            if isinstance(action[i], (np.ndarray, list)):
                # If action[i] is an array or list, take the first element
                action_value = float(action[i][0])
            else:
                # If action[i] is already a scalar, use it directly
                action_value = float(action[i])

            allowed = math.ceil(students * action_value / self.model.total_students)
            allowed_students_per_course.append(allowed)

        updated_infected = estimate_infected_students(self.student_status, allowed_students_per_course, community_risk,
                                                      self.model.number_of_students_per_course())

        self.allowed_students_per_course = allowed_students_per_course
        self.student_status = updated_infected
        self.weekly_infected_students.append(sum(updated_infected))

        self.total_steps += 1

        if self.mode == 'train':
            try:
                self.community_risk = next(self.risk_iterator)
            except StopIteration:
                self.risk_values = self.generate_episode_risk()
                self.risk_iterator = iter(self.risk_values)
                self.community_risk = next(self.risk_iterator)
        elif self.mode == 'eval':
            if hasattr(self, 'community_risk_values') and self.current_time < len(self.community_risk_values):
                self.community_risk = self.community_risk_values[self.current_time]

        self.current_time += 1

    def get_reward(self, alpha: float):
        # Sum all allowed students and infected students across all courses
        total_allowed = sum(self.allowed_students_per_course)
        total_infected = sum(self.student_status)

        # Calculate the reward using the total values
        reward = int(alpha * total_allowed - (1 - alpha) * total_infected)

        return reward

    def is_episode_done(self):
        done = self.current_time >= self.model.get_max_weeks()
        if done:
            logging.info(f"Episode done at time {self.current_time} with max weeks {self.model.get_max_weeks()}")
        return done

    def reset(self):
        self.current_time = 0
        self.allowed_students_per_course = self.model.number_of_students_per_course()

        if self.mode == 'train':
            self.student_status = [random.randint(1, 99) for _ in self.allowed_students_per_course]
            self.risk_values = self.generate_episode_risk()
            self.risk_iterator = iter(self.risk_values)
            self.community_risk = next(self.risk_iterator)
        elif self.mode == 'eval':
            self.student_status = [20 for _ in self.allowed_students_per_course]
            if hasattr(self, 'community_risk_values') and self.community_risk_values:
                self.community_risk = self.community_risk_values[0]

        return self.get_student_status()
