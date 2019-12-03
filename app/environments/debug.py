from app.environments.environment import Environment
from gym import spaces
from math import exp
import numpy as np
from random import random

ACTION_LOW = -100
ACTION_HIGH = 100
OBSERVATION_LOW = -1000
OBSERVATION_HIGH = 1000

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
        self.goal_position = config.goal_position
        # self.epsilon = 10 ** -2
        self.epsilon = 1
        self.state = self.get_initial_state()
        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, shape=(config.action_size,), dtype=np.float)
        self.observation_space = spaces.Box(low=OBSERVATION_LOW, high=OBSERVATION_HIGH, shape=(config.observation_size,), dtype=np.float)

    def get_initial_state(self):
        min_position = self.config.min_start_position
        max_position = self.config.max_start_position
        position = min_position + (max_position - min_position) * random()
        initial_state = np.array([position, self.goal_position], dtype=np.float)
        return initial_state

    def get_next_state(self, action):
        next_state = np.array([self.state[0] + action, self.state[1]], dtype=np.float)
        return next_state

    def get_state(self):
        return self.state

    def is_terminal_state(self, state):
        return abs(state[0] - self.goal_position) < self.epsilon

    def render(self):
        pass

    def reset(self):
        self.state = self.get_initial_state()
        return self.state

    def reward(self, state, action, next_state):
        distance_to_goal = abs(state[0] - self.goal_position)
        next_distance_to_goal = abs(next_state[0] - self.goal_position)
        next_similarity_to_goal = exp(-next_distance_to_goal)
        return (distance_to_goal - next_distance_to_goal) + next_similarity_to_goal

    def step(self, action):
        """
        Finds the next state in the simulated environmet.

        :param action: action performed in current environment
        :return: (next state, reward, terminal, info)
        """
        next_state = self.get_next_state(action)
        reward = self.reward(self.state, action, next_state)
        done = self.is_terminal_state(next_state)
        self.state = next_state
        return next_state, reward, done, {}
