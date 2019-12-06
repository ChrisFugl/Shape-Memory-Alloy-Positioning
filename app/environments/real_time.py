from app.environments.environment import Environment
import numpy as np
import socket
import struct
import time

ACTION_LOW = 0
ACTION_HIGH = 12
OBSERVATION_LOW = -1000
OBSERVATION_HIGH = 1000
POSITION_LOW = 0
POSITION_HIGH = 0.5
TEMP_LOW = 20
TEMP_HIGH = 100
FORCE_LOW = 5
FORCE_HIGH = 10

class RealTimeEnvironment(Environment):
    """
    Real-time environment.

    state:
        * temperature
        * position
        * force
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
        self.action_space = spaces.Box(
            low=ACTION_LOW, 
            high=ACTION_HIGH, 
            dtype=np.float)
        self.observation_space = spaces.Box(
            low=[TEMP_LOW, POSITION_LOW, FORCE_LOW, POSITION_LOW, ACTION_LOW, ACTION_LOW], 
            high=[TEMP_HIGH, POSITION_HIGH, FORCE_HIGH, POSITION_HIGH, ACTION_HIGH, ACTION_HIGH], 
            dtype=np.float)

        if config.port_read == config.port_write:
            client = self.create_connection(config.host, config.port_read)
            self.reader = client
            self.writer = client
        else:
            self.reader = self.create_connection(config.host, config.port_read)
            self.writer = self.create_connection(config.host, config.port_write)

    def is_in_goal(self, position):
        return abs(position - self.goal_position) < self.goal_tolerance

    def create_connection(self, host, port):
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.connect((host, port))
        return connection

    def get_state(self):
        return self.state

    def receive_observation(self):
        values = []
        for _ in range(self.values_per_observation):
            value = self.reader.recv(self.bytes_per_value)
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
        force = self.receive_observation()
        goal_position = self.goal_position
        if self.scale_action and self.pass_scale_interval_to_policy:
            min_voltage, max_voltage = self.get_scaled_action(position)
            state = [temperature, position, force, goal_position, min_voltage, max_voltage]
        else:
            state = [temperature, position, force, goal_position]
        return state, cur_time

    def fetch_state(self):
        observations, cur_time = self.receive_observations()
        return observations, cur_time

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
        next_poisition = next_state[1]
        distance_to_goal = abs(position - self.goal_position)
        next_distance_to_goal = abs(next_position- self.goal_position)
        distance_difference = distance_to_goal - next_distance_to_goal
        if self.is_in_goal(next_position):
            next_similarity_to_goal = 1
        else:
            next_similarity_to_min_start = exp(-abs(next_position - self.min_start_position))
            next_similarity_to_max_start = exp(-abs(next_position - self.max_start_position))
            next_similarity_to_goal = max(next_similarity_to_min_start, next_similarity_to_max_start)
        return distance_difference + next_similarity_to_goal

    def send_action(self, action):
        action_length = self.action_decimal_precision + self.action_digit_precision + 1
        action_string = f'{action:0{action_length}.{self.action_decimal_precision}f}'
        action_encoded = action_string.encode('ascii')
        action_bytes = bytes(action_encoded)
        self.writer.send(action_bytes)

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
        return next_state, reward, terminal

    def get_scaled_action(self, position, action):
        min, max = self.get_action_interval(position)
        # transform to interval (0, 1)
        action_normalized = (action - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW)
        # transform to interval (min, max)
        action_scaled = min + (max - min) * action_normalized
        return action_scaled

    ### Subject to change
    def get_action_interval(self, position):
        return (ACTION_LOW, ACTION_HIGH)
