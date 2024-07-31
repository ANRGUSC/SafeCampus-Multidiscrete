import copy
import logging
import math
import random
from campus_digital_twin import campus_model  # Import the CampusModel from the campus_model module

logging.basicConfig(level=logging.INFO)


class RefactoredCampusState:
    """Class representing the state of the campus."""

    _model = campus_model.CampusModel()

    def __init__(self):
        """Initialize the CampusState object."""
        self._initialized = False
        self._student_status = self._model.number_of_infected_students_per_course()
        self._test_student_status = self._model.test_number_of_infected_students_per_course()
        self._current_time = 0
        self._community_risk = self._model.initial_community_risk()[self._current_time]
        self._allowed_students_per_course = self._model.number_of_students_per_course()[0]
        self._weeks = self._model.initial_community_risk()
        self._states = []
        self._state_transition = []

    def get_total_courses(self):
        return self._model.total_courses()

    def _get_infected_students(self, infected_students, allowed_students_per_course, students_per_course,
                               initial_infection, community_risk):
        """
        Calculate the infected students based on the given parameters.
        (Implementation as per your logic)
        """
        infected_students = []
        for n, f in enumerate(allowed_students_per_course):
            if f == 0:
                correction_factor = 1
                infected_students.append(int(community_risk * students_per_course[n] * correction_factor))

            else:
                asymptomatic_ratio = 0.5
                initial_infection_prob = infected_students[n] / students_per_course[n] * asymptomatic_ratio
                # print("initial infection: ", initial_infection_prob)
                room_capacity = allowed_students_per_course[n]
                infected_prob = calculate_indoor_infection_prob(room_capacity, initial_infection_prob)
                # print("Infected prob: ", infected_prob, " Community risk: ", community_risk)
                total_indoor_infected_allowed = int(infected_prob * allowed_per_course[n])
                total_infected_allowed_outdoor = int(community_risk * allowed_per_course[n])
                total_infected_allowed = min(total_indoor_infected_allowed + total_infected_allowed_outdoor,
                                             allowed_per_course[n])

                infected_students.append(
                    int(total_infected_allowed + community_risk * (students_per_course[n] - allowed_per_course[n])))
        return infected_students

    def get_state(self):
        """
        Get the current state of the campus.
        :return: The current state of the campus.
        """
        return self._get_student_status()

    def set_state(self):
        """
        Set the current state of the campus.
        """
        self._student_status = self._model.number_of_infected_students_per_course().copy()

    def _format_state(self, state):
        """
        Format the state for logging or display.
        :param state: The state to format.
        :return: The formatted state as a string.
        """
        formatted_state = [f"Course {i}: {status}" for i, status in enumerate(state)]
        return ", ".join(formatted_state)

    def _validate_action(self, action):
        """
        Validate the given action input.
        :param action: The action to validate.
        :raise ValueError: If the action is invalid.
        """
        if not isinstance(action, list) or not all(isinstance(i, (int, float)) for i in action):
            raise ValueError("Invalid action input. Action must be a list of numbers.")

    def _calculate_and_update_infected_students(self, action):
        """
        Calculate and update the infected students based on the given action.
        :param action: The action to use for the calculation.
        """
        # ... (Implementation as per your logic)
        pass

    def _reset_student_status(self):
        """
        Reset the student status to its initial state.
        """
        for i in range(len(self._allowed_students_per_course)):
            self._student_status[i] = int(random.random() * self._allowed_students_per_course[i])

    def update_with_action(self, action):
        """
        Update the campus state object with the given action.
        :param action: The action to use for the update.
        """
        self._validate_action(action)
        if self._current_time < self._model.get_max_weeks():
            self._calculate_and_update_infected_students(action)

    def reset(self):
        """
        Reset the current time and student status.
        :return: The reset state of the campus.
        """
        self._current_time = 0
        self._reset_student_status()
        self._community_risk = random.random()
        return self.get_state()


    def _get_student_status(self):
        """
        Get the current student status of the campus.
        :return: The current student status of the campus.
        """
        obs_state = copy.deepcopy(self._student_status)
        obs_state.append(int(self._community_risk * 100))
        return obs_state

    def get_community_risk(self):
        """
        Retrieve the current community risk.
        :return: The current community risk.
        """
        return self._community_risk

    def _set_community_risk(self, risk_range):
        """
        Set the community risk within the provided range.
        :param risk_range: A tuple representing the min and max values for community risk.
        :return: The new community risk value.
        """
        self._community_risk = random.uniform(*risk_range)
        return self._community_risk

    def set_community_risk_high(self):
        """
        Set the community risk to a high value.
        :return: The new community risk value.
        """
        return self._set_community_risk((0.5, 0.9))

    def set_community_risk_low(self):
        """
        Set the community risk to a low value.
        :return: The new community risk value.
        """
        return self._set_community_risk((0.1, 0.4))

    def _calculate_reward(self, alpha, allowed_students, current_infected_students):
        """
        Calculate the reward based on the given parameters.
        :param alpha: The alpha value for reward calculation.
        :param allowed_students: The number of allowed students.
        :param current_infected_students: The number of currently infected students.
        :return: The calculated reward.
        """
        return int(alpha * allowed_students - ((1 - alpha) * current_infected_students))

    def get_reward(self, alpha):
        """
        Calculate the reward given the current state and alpha.
        :param alpha: The alpha value for reward calculation.
        :return: The calculated reward.
        """
        allowed_students = sum(self._allowed_students_per_course)
        current_infected_students = sum(copy.deepcopy(self._student_status))
        return self._calculate_reward(alpha, allowed_students, current_infected_students)

    def update_with_infection_model(self, action):
        """
        Updates the observation with the number of students infected per course.
        :param action: A list with percentage of students to be allowed in a course.
        """
        allowed_students_per_course = []
        students_per_course = self._model.number_of_students_per_course()[0]
        initial_infection = self._model.number_of_infected_students_per_course()

        for i, act in enumerate(action):
            allowed = math.ceil((students_per_course[i] * act) / 100)
            allowed_students_per_course.append(allowed)

        updated_infected = self._get_infected_students(
            self._student_status.copy(),
            allowed_students_per_course,
            students_per_course,
            initial_infection,
            self._community_risk
        )

        self._state_transition.append((self._student_status, updated_infected))
        self._allowed_students_per_course = allowed_students_per_course[:]
        self._student_status = updated_infected[:]

        if self._current_time >= 7:
            self.set_community_risk_low()
        else:
            self.set_community_risk_high()

        self._current_time += 1

    def is_episode_done(self):
        """
        Determines if the episode has reached its termination point.

        Returns:
            bool: True if the current time has reached the maximum allowed weeks, False otherwise.
        """
        return self._current_time == self._model.get_max_weeks()

    def get_test_observations(self):
        test_observation = copy.deepcopy(self._test_student_status)
        test_observation.append(int(self._community_risk * 100))
        return test_observation

    # Reset method and any other remaining methods come here
