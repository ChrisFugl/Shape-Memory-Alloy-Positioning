from app.environments.environment import Environment
from gym import spaces
from math import exp
import numpy as np
import socket
import struct
import time

ACTION_LOW = 0.0
POSITION_LOW = 0
TEMP_LOW = 20
TEMP_HIGH = 100

# define reader and writer globally to ensure
# that only exactly one instance of each will
# exist at any time
reader = None
writer = None

class RealTimeEnvironment(Environment):
    """
    Real-time environment.

    state:
        * temperature
        * position
        * goal position
        * min voltage
        * max voltage

    action:
        * voltage
    """

    name = 'real_time'

    def __init__(self, config):
        """
        :type config: app.config.environments.RealTimeEnvironmentConfig
        """
        global reader
        global writer
        super(RealTimeEnvironment, self).__init__(config)
        self.action_decimal_precision = config.action_decimal_precision
        self.action_digit_precision = config.action_digit_precision
        self.bytes_per_value = config.bytes_per_value
        self.goal_position = config.goal_position
        self.next_state_wait_time = config.next_state_wait_time
        self.values_per_observation = config.values_per_observation
        self.goal_tolerance = config.goal_tolerance
        self.goal_time_tolerance_s = config.goal_time_tolerance_s
        self.scale_action = config.scale_action
        self.pass_scale_interval_to_policy = config.pass_scale_interval_to_policy
        self.max_position = config.max_position
        self.max_exponential_threshold = config.max_exponential_threshold
        self.max_voltage = config.max_voltage
        self.max_threshold_position_ratio = self.max_exponential_threshold / self.max_position
        self.max_position_threshold_distance_exp = exp(abs(self.max_exponential_threshold - self.max_position))
        self.enter_goal_time = None
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=self.max_voltage,
            shape=(config.action_size,),
            dtype=np.float)
        self.observation_space = spaces.Box(
            low=np.array([TEMP_LOW, POSITION_LOW, POSITION_LOW, ACTION_LOW, ACTION_LOW]),
            high=np.array([TEMP_HIGH, self.max_position, self.max_position, self.max_voltage, self.max_voltage]),
            dtype=np.float)

        if reader is None or writer is None:
            if config.port_read == config.port_write:
                client = self.create_connection(config.host, config.port_read)
                reader = client
                writer = client
            else:
                reader = self.create_connection(config.host, config.port_read)
                writer = self.create_connection(config.host, config.port_write)

    def is_in_goal(self, position):
        return abs(position - self.goal_position) < self.goal_tolerance

    def create_connection(self, host, port):
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.connect((host, port))
        return connection

    def get_state(self):
        return self.state

    def receive_observation(self):
        global reader
        values = []
        for _ in range(self.values_per_observation):
            value = reader.recv(self.bytes_per_value)
            decoded_value = value.decode('ascii')
            float_value = float(decoded_value)
            values.append(float_value)
        return np.mean(values)

    def receive_observations(self):
        cur_time = time.time()
        temperature = self.receive_observation()
        position = self.receive_observation()
        if self.enter_goal_time is None and self.is_in_goal(position):
            self.enter_goal_time = cur_time
        elif self.enter_goal_time is not None and not self.is_in_goal(position):
            self.enter_goal_time = None
        goal_position = self.goal_position
        if self.scale_action and self.pass_scale_interval_to_policy:
            min_voltage, max_voltage = self.get_action_interval(position)
            state = [temperature, position, goal_position, min_voltage, max_voltage]
        else:
            state = [temperature, position, goal_position]
        return state, cur_time

    def fetch_state(self):
        observations, cur_time = self.receive_observations()
        return np.array(observations, dtype=np.float), cur_time

    def is_terminal_state(self, state, state_time):
        return state_time is not None and state_time >= self.goal_time_tolerance_s

    def reset(self):
        self.enter_goal_time = None
        self.state, _ = self.fetch_state()
        return self.state

    def render(self):
        pass

    def reward(self, state, action, next_state):
        position = state[1]
        next_position = next_state[1]
        distance_to_goal = abs(position - self.goal_position)
        next_distance_to_goal = abs(next_position- self.goal_position)
        distance_difference = distance_to_goal - next_distance_to_goal
        if self.is_in_goal(next_position):
            next_similarity_to_goal = 1
        else:
            next_similarity_to_min_goal = exp(-abs(next_position - (self.goal_position - self.goal_tolerance)))
            next_similarity_to_max_goal = exp(-abs(next_position - (self.goal_position + self.goal_tolerance)))
            next_similarity_to_goal = max(next_similarity_to_min_goal, next_similarity_to_max_goal)
        return distance_difference + next_similarity_to_goal

    def send_action(self, action):
        global writer
        action_length = self.action_decimal_precision + self.action_digit_precision + 1
        action_string = f'{action:0{action_length}.{self.action_decimal_precision}f}'
        action_encoded = action_string.encode('ascii')
        action_bytes = bytes(action_encoded)
        writer.send(action_bytes)

    def step(self, action):
        """
        Finds the next state in the simulated environmet.

        :param action: action performed in current environment
        :return: (next state, reward)
        """
        if self.scale_action:
            action = self.get_scaled_action(self.state[1], action)
        self.send_action(action[0])
        # it may be necessary to wait a while in order for the action
        # to have an observable effect
        if self.next_state_wait_time is not None:
            time.sleep(self.next_state_wait_time)
        next_state, next_state_time = self.fetch_state()
        reward = self.reward(self.state, action, next_state)
        terminal = self.is_terminal_state(next_state, next_state_time)
        self.state = next_state
        return next_state, reward, terminal, {}

    def get_scaled_action(self, position, action):
        min, max = self.get_action_interval(position)
        # transform to interval (0, 1)
        action_normalized = (action - ACTION_LOW) / (self.max_voltage - ACTION_LOW)
        # transform to interval (min, max)
        action_scaled = min + (max - min) * action_normalized
        return action_scaled

    def get_action_interval(self, position):
        """
        Compute min and max voltage.

        Minimum should always be at 0.
        Maximum is linear interpolated as a function of the position until it reaches a configurable threshold.
        It interpolates exponentially from the threshold and until the maximum position.
        Maximum voltage is 0 in the (theoretical) case that position is greater than maximum position.
        """
        if position <= self.max_exponential_threshold:
            max_position_ratio = position / self.max_position
            max_voltage = ACTION_LOW + (self.max_voltage - ACTION_LOW) * (1.0 - max_position_ratio)
        elif position >= self.max_position:
            max_voltage = 0.0
        else:
            max_voltage_at_threshold = ACTION_LOW + (self.max_voltage - ACTION_LOW) * (1.0 - self.max_threshold_position_ratio)
            distance_to_max_position = abs(position - self.max_position)
            distance_ratio = exp(distance_to_max_position) / self.max_position_threshold_distance_exp
            max_voltage = max_voltage_at_threshold * distance_ratio
        return ACTION_LOW, max_voltage
