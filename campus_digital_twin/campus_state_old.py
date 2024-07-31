import math
import copy
import random
from models.infection_model import get_infected_students_sir
from models.infection_model import get_infected_students_apprx_si
seed_value = 500
random.seed(seed_value)


def map_value_to_range(old_value, old_min=0., old_max=0.1, new_min=0, new_max=100):
    """Map a value from the old range to the new range."""
    return (old_value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

class Simulation:
    def __init__(self, model):
        self.current_time = 0
        self.model = model
        # Handle multiple courses dynamically
        self.allowed_students_per_course =[]
        self.student_status = model.initial_infection
        self.state_transition = []
        self.community_risk = random.random()
        self.weekly_infected_students = []
        self.allowed = []
        self.infected = []
        self.const_1 = random.uniform(.4,.6)
        self.const_2 = random.uniform(.4,.6)
        print("initial infected students: ", self.student_status) #debug check

    def set_community_risk_high(self):
        self.community_risk = random.uniform(0.5, 1.0)
        return self.community_risk

    def set_community_risk_low(self):
        self.community_risk = random.uniform(0.0, 0.5)
        return self.community_risk

    def get_student_status(self):
        obs_state = copy.deepcopy(self.student_status)
        # fixme: this is a hack to get the community risk value
        # fixme: this should be removed once the community risk is added to the student_status
        # obs_state.append(int(map_value_to_range(self.community_risk)))
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
            for i, students in enumerate(self. model.number_of_students_per_course())
        ]

        updated_infected = get_infected_students_sir(self.student_status, allowed_students_per_course, community_risk)
        self.allowed_students_per_course = allowed_students_per_course
        self.student_status = updated_infected
        self.weekly_infected_students.append(sum(updated_infected))

        if self.current_time <= int(self.model.get_max_weeks()/2):

            self.set_community_risk_low()
        else:
            self.set_community_risk_high()

        self.current_time += 1

    def get_reward(self, alpha: float):

        current_infected_students = sum(self.student_status)

        allowed_students = sum(self.allowed_students_per_course)
        # Calculating the reward, keeping it as float for more precision
        return alpha * allowed_students - ((1 - alpha) * current_infected_students)

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
        self.student_status = [random.randint(0, 30) for _ in self.allowed_students_per_course]
        self.community_risk = random.uniform(0.0, 1.0)
        return self.get_student_status()





