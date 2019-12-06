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

    name = 'debug'

    def __init__(self, config):
        """
        :type config: app.config.environments.DebugEnvironmentConfig
        """
        super(DebugEnvironment, self).__init__(config)
        self.goal_position = config.goal_position
        self.goal_tolerance = config.goal_tolerance
        self.min_start_position = config.min_start_position
        self.max_start_position = config.max_start_position
        self.pass_scale_interval_to_policy = config.pass_scale_interval_to_policy
        self.scale_action = config.scale_action
        self.state = self.get_initial_state()
        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, shape=(config.action_size,), dtype=np.float)
        self.observation_space = spaces.Box(low=OBSERVATION_LOW, high=OBSERVATION_HIGH, shape=(config.observation_size,), dtype=np.float)

    def get_initial_state(self):
        position = self.min_start_position + (self.max_start_position - self.min_start_position) * random()
        if self.scale_action and self.pass_scale_interval_to_policy:
            min, max = self.get_action_interval(position)
            initial_state = [position, self.goal_position, min, max]
        else:
            initial_state = [position, self.goal_position]
        return np.array(initial_state, dtype=np.float)

    def get_next_state(self, state, action):
        next_position = state[0] + action
        if self.scale_action and self.pass_scale_interval_to_policy:
            min, max = self.get_action_interval(next_position)
            next_state = [next_position, state[1], min, max]
        else:
            next_state = [next_position, state[1]]
        return np.array(next_state, dtype=np.float)

    def get_state(self):
        return self.state

    def is_terminal_state(self, state):
        return abs(state[0] - self.goal_position) < self.goal_tolerance

    def render(self):
        pass

    def reset(self):
        self.state = self.get_initial_state()
        return self.state

    def reward(self, state, action, next_state):
        distance_to_goal = abs(state[0] - self.goal_position)
        next_distance_to_goal = abs(next_state[0] - self.goal_position)
        distance_difference = distance_to_goal - next_distance_to_goal
        if self.is_terminal_state(next_state):
            next_similarity_to_goal = 1
        else:
            next_similarity_to_min_start = exp(-abs(next_state[0] - self.min_start_position))
            next_similarity_to_max_start = exp(-abs(next_state[0] - self.max_start_position))
            next_similarity_to_goal = max(next_similarity_to_min_start, next_similarity_to_max_start)
        return distance_difference + next_similarity_to_goal

    def step(self, action):
        """
        Finds the next state in the simulated environmet.
        :param action: action performed in current environment
        :return: (next state, reward, terminal, info)
        """
        if self.scale_action:
            action = self.get_scaled_action(self.state[0], action)
        next_state = self.get_next_state(self.state, action)
        reward = self.reward(self.state, action, next_state)
        done = self.is_terminal_state(next_state)
        self.state = next_state
        return next_state, reward, done, {}

    def get_scaled_action(self, position, action):
        min, max = self.get_action_interval(position)
        # transform to interval (0, 1)
        action_normalized = (action + 1.0) / 2.0
        # transform to interval (min, max)
        action_scaled = min + (max - min) * action_normalized
        return action_scaled

    def get_action_interval(self, position):
        distance_to_goal = abs(position - self.goal_position)
        a = min(0.33, random())
        b = 1.0 - a
        lowest = a * distance_to_goal
        highest = b * distance_to_goal
        if position < self.goal_position:
            return (lowest, highest)
        else:
            return (-highest, -lowest)