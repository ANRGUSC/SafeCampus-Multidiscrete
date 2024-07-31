import math
import copy
import random
from enum import Enum
from epidemic_models.analyze_models import estimate_infected_students_sir, estimate_infected_students, get_infected_students
random.seed(500)

# 100HIGH_COMMUNITY_RISK = 0.7
# LOW_COMMUNITY_RISK = 0.3

# Enum for Community Risk
class CommunityRisk(Enum):
    LOW = random.uniform(0.01, 0.055)
    HIGH = random.uniform(0.055, 0.1)

def map_value_to_range(old_value, old_min=0.01, old_max=0.1, new_min=0, new_max=100):
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
        # print("action: ", action) #debug check
        # print("students per course: ", self.model.number_of_students_per_course()) #debug check

        allowed_students_per_course = [
            math.ceil(students * action[i] / self.model.total_students)
            for i, students in enumerate(self. model.number_of_students_per_course())
        ]
        initial_infection = self.model.get_initial_infection()
        # updated_infected = get_infected_students(self.student_status, allowed_students_per_course,
        #                       self.model.number_of_students_per_course(), initial_infection, community_risk)
        updated_infected = estimate_infected_students(self.student_status, allowed_students_per_course, community_risk,
                                                      self.model.number_of_students_per_course())

        # print("updated infected students: ", updated_infected) #debug check

        # self.state_transition.append((self.student_status, updated_infected))
        self.allowed_students_per_course = allowed_students_per_course
        self.student_status = updated_infected
        # print("allowed students per course: ", self.allowed_students_per_course) #debug check
        # print("student status: ", self.student_status) #debug check
        self.weekly_infected_students.append(sum(updated_infected))
        # print("Initial Infected: ", self.model.get_initial_infection())
        # print("Updated Infected: ", updated_infected)
        # print("Normalized Infected: ", normalized_infected)

        # self.community_risk = random.uniform(0.01, 0.1)

        if self.current_time >= int(self.model.max_weeks/2):
            self.set_community_risk_low()
            # self.community_risk = self.community_risk * self.set_community_risk_low() * random.uniform(0.0, 0.1) + self.community_risk
        else:
            self.set_community_risk_high()
            # self.community_risk = self.community_risk * self.set_community_risk_high() * random.uniform(0.0, 0.1) + self.community_risk

        self.current_time += 1


    def get_reward(self, alpha: float):
        rewards = []
        for i in range(len(self.student_status)):
            reward = int(alpha * self.allowed_students_per_course[i] - ((1 - alpha) * self.student_status[i]))
            rewards.append(reward)
        return sum(rewards)
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
        self.student_status = [random.randint(1, 99) for _ in self.allowed_students_per_course]
        self.community_risk = random.uniform(0.0, 1.0)
        # print("Resetting the state...: ", self.student_status, self.community_risk) #debug check
        return self.get_student_status()





