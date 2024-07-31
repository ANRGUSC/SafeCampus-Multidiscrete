"""This class implements the campus_digital_twin environment

The campus environment is composed of the following:
   - Students taking courses.
   - Courses offered by the campus.
   - Community risk provided to the campus every week.

   Agents control the number of students allowed to sit on a course per week.
   Observations consists of an ordered list that contains the number of the
   infected students and the community risk value. Every week the agent proposes what
   percentage of students to allow on campus.

   Actions consists of 3 levels for each course. These levels correspond to:
    - 0%: schedule class online
    - 50%: schedule 50% of the class online
    - 100%: schedule the class offline

   An episode ends after 15 steps (Each step represents a week).
   We assume an episode represents a semester.

"""
import pygame
import gymnasium as gym
# import gym
from campus_digital_twin import cs_test, test_campus_model
import numpy as np
import json
import logging
logging.basicConfig(filename="run.txt", level=logging.INFO)


def visualize_classroom(weekly_infected_students):
    # Define display dimensions and colors
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    white = (255, 255, 255)
    red = (255, 0, 0)

    # Set the title of the window
    pygame.display.set_caption('Classroom Visualization')

    # Main Loop
    running = True
    week_index = 0
    clock = pygame.time.Clock()

    while running:
        screen.fill(white)

        # Display infected students for the current week
        infected_students_count = weekly_infected_students[week_index]
        radius = 5
        start_position = (50, 50)  # you can change the start position
        distance_between_circles = 10  # distance between centers of two consecutive circles

        for i in range(infected_students_count):
            x_position = start_position[0] + i * (radius * 2 + distance_between_circles)
            y_position = start_position[1]
            pygame.draw.circle(screen, red, (x_position, y_position), radius)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    week_index = min(week_index + 1, len(weekly_infected_students) - 1)
                elif event.key == pygame.K_LEFT:
                    week_index = max(week_index - 1, 0)

        # Update the display and control the frame rate
        pygame.display.flip()
        clock.tick(30)



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
    Converts a list of action or state values to a list of discrete values [0, 1, 2].

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
    discrete_actions_list = [get_discrete_value(value) for value in action_or_state]

    return discrete_actions_list


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
    """
        Defines a Gym environment representing a campus scenario where agents control
        the number of students allowed to sit on a course per week, considering the
        risk of infection and the reward is based on the number of allowed students.

        Observation:
            Type: Multidiscrete([0, 1 ..., n+1]) where n is the number of courses, and
            the last item is the community risk value.
            Example observation: [20, 34, 20, 0.5]

        Actions:
            Type: Multidiscrete([0, 1 ... n]) where n is the number of courses.
            Example action: [0, 1, 1]

        Reward:
            Reward is a scalar value returned from the campus environment.
            A high reward corresponds to an increase in the number of allowed students.

        Episode Termination:
            The episode terminates after n steps, representing the duration of campus operation.
        """
    metadata = {'render.modes': ['bot']}

    def __init__(self):
        pygame.init()

        # Initialize a new campus state object
        self.campus_state = cs_test.Simulation(model=test_campus_model.CampusModel())
        total_courses = test_campus_model.CampusModel().num_courses

        # Define action and observation spaces
        num_infection_levels = 10
        num_occupancy_levels = 3

        self.action_space = gym.spaces.MultiDiscrete([num_occupancy_levels] * total_courses)
        self.observation_space = gym.spaces.MultiDiscrete([num_infection_levels] * (total_courses + 1))



    def step(self, action):
        """
            Execute one time step within the environment.
        """

        # Extract alpha from the list of action and update the campus state with the action
        alpha = action[-1]
        action.pop()
        self.campus_state.update_with_action(action)

        # Obtain observation, reward, and check if the episode is done
        observation = np.array(convert_actions_to_discrete(self.campus_state.get_student_status()))
        reward = self.campus_state.get_reward(alpha)
        done = self.campus_state.is_episode_done()
        # done = self.campus_state.current_time == self.campus_state.model.get_max_weeks()
        info = {
            "allowed": self.campus_state.allowed_students_per_course,
            "infected": self.campus_state.student_status,
            "reward": reward
        }

        return observation, reward, done, False, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Returns:    observation (object): the initial observation.
        """
        state = self.campus_state.reset()
        logging.info(f"reset state: {state}")
        discrete_state = convert_actions_to_discrete(state)

        return np.array(discrete_state), {}


    def render(self, mode='bot'):
        """
        Render the environment's state.
        """
        weekly_infected_students = self.campus_state.weekly_infected_students
        visualize_classroom(weekly_infected_students)
        return None

    def close(self):
        pygame.quit()
