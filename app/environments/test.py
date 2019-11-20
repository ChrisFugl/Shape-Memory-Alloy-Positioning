from app.environments.environment import Environment
from math import exp
import numpy as np

class TestEnvironment(Environment):
    """
    Environment used for testing purposes.

    state: 1D ndarray of size 1
    action: 1D ndarray of size 1
    """

    def __init__(self, initial_state, final_state):
        self.initial_state = initial_state
        self.final_state = final_state
        super(TestEnvironment, self).__init__(None)

    def get_initial_state(self, options):
        return np.array([self.initial_state])

    def get_next_state(self, action):
        return self.state + action[0]

    def is_terminal_state(self):
        return self.state[0] == self.final_state

    def reward(self, state, action, next_state):
        distance_next = abs(self.final_state - next_state[0])
        distance_current = abs(self.final_state - state[0])
        return distance_current - distance_next
