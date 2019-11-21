from app.environments.environment import Environment
from math import exp
import numpy as np

class TestEnvironment(Environment):
    """
    Environment used for testing purposes.

    state: 1D ndarray of size 1
    action: 1D ndarray of size 1
    """

    def __init__(self, config):
        """
        :type config: app.config.environments.TestEnvironmentConfig
        """
        self.initial_state = config.initial_state
        self.final_state = config.final_state
        self.config = config
        super(TestEnvironment, self).__init__(config)

    def get_initial_state(self):
        return np.array([self.initial_state])

    def get_next_state(self, action):
        return self.state + action

    def is_terminal_state(self, state):
        return state[0] == self.final_state

    def reset(self):
        # TODO
        pass

    def reward(self, state, action, next_state):
        distance_next = abs(self.final_state - next_state[0])
        distance_current = abs(self.final_state - state[0])
        return distance_current - distance_next
