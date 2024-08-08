import math
import copy
import random
import csv
import logging
from epidemic_models.analyze_models import estimate_infected_students

random.seed(500)


class Simulation:
    def __init__(self, model, read_community_risk_from_csv=False, csv_path=None):
        self.current_time = 0
        self.model = model
        self.allowed_students_per_course = []
        self.student_status = [20] * model.num_courses
        self.state_transition = []
        self.weekly_infected_students = []
        self.allowed = []
        self.infected = []
        self.community_risk = random.random()  # Default random value

        if read_community_risk_from_csv and csv_path:
            self.read_community_risk_from_csv(csv_path)
        else:
            self.community_risk_values = [self.community_risk] * self.model.get_max_weeks()

        # Logging max_weeks from model
        logging.info(f"Simulation initialized with max weeks: {self.model.get_max_weeks()}")

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
        self.community_risk = random.uniform(0.5, 1.0)
        return self.community_risk

    def set_community_risk_low(self):
        self.community_risk = random.uniform(0.0, 0.5)
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
        # print("student_status: ", self.student_status)
        # print('initial_infection: ', self.model.get_initial_infection())
        allowed_students_per_course = [
            math.ceil(students * action[i] / self.model.total_students)
            for i, students in enumerate(self.model.number_of_students_per_course())
        ]
        initial_infection = self.model.get_initial_infection()
        updated_infected = estimate_infected_students(self.student_status, allowed_students_per_course, community_risk,
                                                      self.model.number_of_students_per_course())

        self.allowed_students_per_course = allowed_students_per_course
        self.student_status = updated_infected
        self.weekly_infected_students.append(sum(updated_infected))

        if self.current_time >= int(self.model.max_weeks / 2):
            self.set_community_risk_low()
        else:
            self.set_community_risk_high()


        self.current_time += 1
        # For evaluation purposes
        # if hasattr(self, 'community_risk_values') and self.current_time < len(self.community_risk_values):
        #     self.community_risk = self.community_risk_values[self.current_time]
        # logging.info(
        #     f"Step {self.current_time}: community_risk={self.community_risk}, infected={self.student_status}, allowed={self.allowed_students_per_course}")

    def get_reward(self, alpha: float):
        rewards = []
        for i in range(len(self.student_status)):
            reward = int(alpha * self.allowed_students_per_course[i] - ((1 - alpha) * self.student_status[i]))
            rewards.append(reward)
        return sum(rewards)

    def is_episode_done(self):
        done = self.current_time >= self.model.get_max_weeks()
        if done:
            logging.info(f"Episode done at time {self.current_time} with max weeks {self.model.get_max_weeks()}")
        return done

    def reset(self):
        self.current_time = 0
        self.allowed_students_per_course = self.model.number_of_students_per_course()
        self.student_status = [random.randint(1, 99) for _ in self.allowed_students_per_course] # for training
        # self.student_status = [20 for _ in self.allowed_students_per_course] # for testing
        if hasattr(self, 'community_risk_values') and self.community_risk_values:
            self.community_risk = self.community_risk_values[0]
        else:
            self.community_risk = random.uniform(0.0, 1.0)
        return self.get_student_status()  # Call the method to return the actual state
