import math
import copy
import random
from enum import Enum
from scipy.stats import binom
import numpy as np
from campus_digital_twin import campus_model as cm
from models.infection_model import get_infected_students


HIGH_COMMUNITY_RISK = 0.7
LOW_COMMUNITY_RISK = 0.3

# Enum for Community Risk
class CommunityRisk(Enum):
    HIGH = HIGH_COMMUNITY_RISK
    LOW = LOW_COMMUNITY_RISK


class Simulation:
    def __init__(self, model):
        self.current_time = 0
        self.model = model
        # Handle multiple courses dynamically
        self.allowed_students_per_course = model.number_of_students_per_course()
        print(self.allowed_students_per_course)  # Debug: Check the value
        # self.student_status = [int(random.random() * students) for students in self.allowed_students_per_course]

        # Initialize student_status for each course
        self.student_status = [[int(random.random() * students)
                                for _ in range(students)]
                               for students in self.allowed_students_per_course]
        # self.allowed_students_per_course = model.number_of_students_per_course()[0]
        # self.student_status = [int(random.random() * students) for students in self.allowed_students_per_course]
        self.state_transition = []
        self.community_risk = random.random()
        self.weekly_infected_students = []

    def set_community_risk_high(self):
        self.community_risk = CommunityRisk.HIGH.value

    def set_community_risk_low(self):
        self.community_risk = CommunityRisk.LOW.value

    def get_student_status(self):
        obs_state = copy.deepcopy(self.student_status)
        # fixme: this is a hack to get the community risk value
        # fixme: this should be removed once the community risk is added to the student_status
        obs_state.append(int(self.community_risk * 100))
        return obs_state

    def update_with_action(self, action):
        """Updates the campus state object with action.
        Args:
             action: A list with percentage of students to allow for each course.
        Returns:
            None
        """
        if self.current_time < self.model.get_max_weeks():
            self.apply_action(action, self.community_risk)

        return None

    def apply_action(self, action: list, community_risk: float):
        allowed_students_per_course = [
            math.ceil(students * action[i] / 100)
            for i, students in enumerate(self.allowed_students_per_course)
        ]
        initial_infection = self.model.get_initial_infection()
        updated_infected = get_infected_students(self.student_status, allowed_students_per_course,
                              self.model.number_of_students_per_course(), initial_infection, community_risk)

        self.state_transition.append((self.student_status, updated_infected))
        self.allowed_students_per_course = allowed_students_per_course
        self.student_status = updated_infected
        self.weekly_infected_students.append(sum(updated_infected))

        if self.current_time >= 7:
            self.set_community_risk_low()
        else:
            self.set_community_risk_high()

        self.current_time += 1

    def get_reward(self, alpha: float):
        current_infected_students = sum(self.student_status)
        allowed_students = sum(self.allowed_students_per_course)
        return int(alpha * allowed_students - ((1 - alpha) * current_infected_students))

    def is_episode_done(self):
        """
        Determines if the episode has reached its termination point.

        Returns:
            bool: True if the current time has reached the maximum allowed weeks, False otherwise.
        """
        return self.current_time == self.model.get_max_weeks()

    def reset(self):
        self.current_time = 0
        self.allowed_students_per_course = self.model.number_of_students_per_course()
        self.student_status = [int(random.random() * students) for students in self.allowed_students_per_course]
        self.community_risk = random.random()
        return self.get_student_status()





