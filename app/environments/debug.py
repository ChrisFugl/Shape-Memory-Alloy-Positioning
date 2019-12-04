from app.environments.environment import Environment
from math import exp
import numpy as np
from random import random

class DebugEnvironment(Environment):
    """
    Environment used for debugging.
    """

    def __init__(self, config):
        """
        :type config: app.config.environments.DebugEnvironmentConfig
        """
        super(DebugEnvironment, self).__init__(config)
        self.config = config
        self.epsilon = 10 ** -9

    def get_initial_state(self):
        min_position = self.config.min_start_position
        max_position = self.config.max_start_position
        # position = min_position + (max_position - min_position) * random()
        # print("Current position: " + str(position))
        position = 2
        return np.array([position])

    def get_next_state(self, action):
        return self.state + action

    def get_state(self):
        return self.state

    def is_terminal_state(self, state):
        return abs(state[0] - self.config.goal_position) < self.epsilon

    def reset(self):
        self.state = self.get_initial_state()

    # def reward(self, state, action, next_state):
    #     distance_to_goal = abs(next_state[0] - self.config.goal_position)
    #     similarity_to_goal = exp(-distance_to_goal)
    #     return similarity_to_goal

    def reward(self, state, action, next_state):
        new_distance_to_goal = abs(next_state[0] - self.config.goal_position)
        old_distance_to_goal = abs(state[0] - self.config.goal_position)
        distance = old_distance_to_goal - new_distance_to_goal 
        direction = 0
        # ext = 1
        ext = 10
        gam = 0.1
        if distance < 0:
            direction = 1
        if direction == 1:
            # similarity_to_goal = exp((ext - gam * new_distance_to_goal)) ### When future position is closer to the target
            similarity_to_goal = exp(-gam * new_distance_to_goal)
        else:
            # similarity_to_goal = exp((- ext - gam * new_distance_to_goal)) ### When future position is further from the target
            similarity_to_goal = -exp(gam * new_distance_to_goal)
        return similarity_to_goal

    def step(self, action):
        """
        Finds the next state in the simulated environmet.

        :param action: action performed in current environment
        :return: (next state, reward)
        """
        next_state = self.get_next_state(action)
        reward = self.reward(self.state, action, next_state)
        terminal = self.is_terminal_state(next_state)
        self.state = next_state
        return next_state, reward, terminal
