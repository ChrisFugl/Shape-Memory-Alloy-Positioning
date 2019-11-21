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
        self.config = config
        self.epsilon = 10 ** -9
        super(DebugEnvironment, self).__init__(config)

    def get_initial_state(self):
        min_position = self.config.min_start_position
        max_position = self.config.max_start_position
        position = min_position + (max_position - min_position) * random()
        return np.array([position])

    def get_next_state(self, action):
        return self.state + action

    def is_terminal_state(self, state):
        return abs(state[0] - self.config.goal_position) < self.epsilon

    def reset(self):
        self.state = self.get_initial_state()

    def reward(self, state, action, next_state):
        distance_to_goal = abs(next_state[0] - self.config.goal_position)
        similarity_to_goal = exp(-distance_to_goal)
        return similarity_to_goal
