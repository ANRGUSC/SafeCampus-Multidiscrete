"""
This class implements the campus_digital_twin environment
"""
import gymnasium as gym
from campus_digital_twin import campus_model, campus_state
import numpy as np
import logging

def get_discrete_value(number):
    """
    Converts a given number to a discrete value based on its range.

    Parameters:
    number (int or float): The input number to be converted to a discrete value.

    Returns:
    int: A discrete value representing the range in which the input number falls.
         It returns a value between 0 and 9, inclusive.

    Example:
    get_discrete_value(25) returns 2
    get_discrete_value(99) returns 9
    """

    # Ensure the number is within the range [0, 100]
    number = min(99, max(0, number))

    # Perform integer division by 10 to get the discrete value
    # This will also ensure that the returned value is an integer
    return number // 10

def convert_actions_to_discrete(action_or_state):
    """
    Converts a list of state values to a list of discrete values [0, 1, 2].

    This function applies the get_discrete_value function to each element in the input list,
    converting them to discrete values and returning the new list of discrete values.

    Parameters:
    action_or_state (list of int or float): A list containing action or state values to be converted.

    Returns:
    list of int: A list containing the converted discrete values.

    Example:
    convert_actions_to_discrete([15, 25, 35]) returns [1, 2, 3]
    """

    # Use list comprehension to apply get_discrete_value to each element in action_or_state
    return [get_discrete_value(value) for value in action_or_state]

def disc_conv_action(discrete_actions_list):
    """
    Converts a list of discrete action values to a list of actions in the range [0, 100].

    Parameters:
    discrete_actions_list (list of int): A list containing discrete action values.

    Returns:
    list of int: A list containing converted action values in the range [0, 100].

    Example:
    disc_conv_action([0, 1, 2]) returns [0, 50, 100]
    """

    # Use list comprehension to convert each discrete action value
    # in discrete_actions_list to the range [0, 100]
    return [(int)(val * 50) for val in discrete_actions_list]

class CampusGymEnv(gym.Env):
    metadata = {'render.modes': ['bot']}

    def __init__(self, read_community_risk_from_csv=False, csv_path=None, algorithm='q_learning', mode='train'):
        self.campus_state = campus_state.Simulation(
            model=campus_model.CampusModel(read_weeks_from_csv=read_community_risk_from_csv, csv_path=csv_path),
            read_community_risk_from_csv=read_community_risk_from_csv,
            csv_path=csv_path,
            mode=mode
        )
        self.students_per_course = self.campus_state.model.number_of_students_per_course()
        total_courses = len(self.students_per_course)

        num_infection_levels = 10
        num_occupancy_levels = 3

        self.action_space = gym.spaces.MultiDiscrete([num_occupancy_levels] * total_courses)
        self.observation_space = gym.spaces.MultiDiscrete([num_infection_levels] * (total_courses + 1))

        self.algorithm = algorithm

        # Log the max weeks
        logging.info(f"Environment initialized with max weeks: {self.campus_state.model.get_max_weeks()}")

    def step(self, action):
        # print(f"Algorithm: {self.algorithm}, Action Type: {type(action)}")
        if self.algorithm == 'q_learning':
            # For q_learning, action is expected to be a list where the last element is alpha
            alpha = action.pop()

            # Update campus state with the raw action values directly (since actions can now be arbitrary)
            self.campus_state.update_with_action(action)

            # The observation should reflect the state after the action is taken
            observation = np.array(self.campus_state.get_student_status())

        elif self.algorithm == 'dqn':
            # For dqn, action is a tuple with the first element as actions and the second as alpha
            actions, alpha = action[0], action[1]  # This unpacks the list into actions and alpha
            if not isinstance(actions, list):
                actions = [actions]  # Ensure actions is a list

            # Update campus state with the raw actions directly
            self.campus_state.update_with_action(actions)

            # The observation for DQN does not need discretization
            observation = np.array(self.campus_state.get_student_status())

        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        # Calculate the reward using the post-action state
        reward = self.campus_state.get_reward(alpha)

        # Determine if the episode is done
        done = self.campus_state.is_episode_done()

        info = {
            "allowed": self.campus_state.allowed_students_per_course,
            "infected": self.campus_state.student_status,
            "community_risk": self.campus_state.community_risk,
            "reward": reward,
            "continuous_state": self.campus_state.get_student_status()
        }
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        # Seed the environment's RNG
        super().reset(seed=seed)

        # Reset the campus state
        state = self.campus_state.reset()

        if self.algorithm == 'q_learning':
            discrete_state = convert_actions_to_discrete(state)
            return np.array(discrete_state), {}
        elif self.algorithm == 'dqn':
            return np.array(state), {}
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def render(self, mode='bot'):
        weekly_infected_students = int(sum(self.campus_state.weekly_infected_students)) / len(self.campus_state.weekly_infected_students)
        allowed_students_per_course = self.campus_state.allowed_students_per_course
        print("weekly_infected_students: ", weekly_infected_students, "allowed_students_per_course: ", allowed_students_per_course)
        return None