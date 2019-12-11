from app.environments.environment import Environment
from gym import spaces
from math import exp
import numpy as np
import random
from scipy import stats
import socket
import struct
import time

ACTION_LOW = -1.0
ACTION_HIGH = 1.0
POSITION_LOW = 0
TEMP_LOW = 0.0
TIME_LOW = 0.0
TIME_HIGH = 30.0

MIN_VOLTAGE = 0.0

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
        * temperature change
        * position
        * position change
        * time since last state (seconds)
        * goal position
        * min voltage
        * max voltage

    min and max voltage are optional and are only added when it is configured
    by setting scale_action and pass_scale_interval_to_policy as true

    action:
        * voltage
    """

    name = 'real_time'

    def __init__(self, config):
        """
        :type config: app.config.environments.RealTimeEnvironmentConfig
        """
        super(RealTimeEnvironment, self).__init__(config)
        self._action_decimal_precision = config.action_decimal_precision
        self._action_digit_precision = config.action_digit_precision
        self._bytes_per_value = config.bytes_per_value
        self._goal_position = config.goal_position
        self._next_state_wait_time = config.next_state_wait_time
        self._values_per_observation = config.values_per_observation
        self._reset_tolerance = config.reset_tolerance

        # goal variables
        self._goal_type = config.goal_type
        self._goal_start_position = config.goal_position
        self._goal_min = config.goal_min
        self._goal_max = config.goal_max
        self._goal_tolerance = config.goal_tolerance
        self._goal_time_tolerance_s = config.goal_time_tolerance_s

        # reward variables
        self._reward_trunc_min = config.reward_trunc_min
        self._reward_trunc_max = config.reward_trunc_max
        self._reward_std = config.reward_std
        self._reward_gaussian = None
        self._reward_gaussian_max = None

        self._enter_goal_time = None
        self._goal_position = None
        self._state = None
        self._state_timestep = None

        self._setup_action_scaling(config)
        self._setup_connections(config)
        self.action_space = self._get_action_space(config)
        self.observation_space = self._get_observation_space(config)

    def _setup_connections(self, config):
        global reader
        global writer
        if reader is None or writer is None:
            if config.port_read == config.port_write:
                client = self.create_connection(config.host, config.port_read)
                reader = client
                writer = client
            else:
                reader = self.create_connection(config.host, config.port_read)
                writer = self.create_connection(config.host, config.port_write)

    def _setup_action_scaling(self, config):
        self._scale_action = config.scale_action
        self._pass_scale_interval = config.scale_action and config.pass_scale_interval_to_policy
        self._max_position = config.max_position
        self._max_position_limit = config.max_position + config.reset_tolerance
        self._max_voltage = config.max_voltage
        self._max_linear_threshold_position = config.max_linear_threshold_position
        self._max_linear_threshold_voltage = config.max_linear_threshold_voltage
        self._max_temperature = config.max_temperature
        self._line_slope = (self._max_linear_threshold_voltage - self._max_voltage) / (self._max_linear_threshold_position - POSITION_LOW)
        self._line_intersection = config.max_voltage
        self._max_position_threshold_distance = abs(self._max_linear_threshold_position - self._max_position)

    def _get_action_space(self, config):
        return spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, shape=(config.action_size,), dtype=np.float)

    def _get_observation_space(self, config):
        temp_high = config.max_temperature + 10.0
        observation_low = [TEMP_LOW, -temp_high, POSITION_LOW, -self._max_position_limit, TIME_LOW, POSITION_LOW]
        observation_high = [temp_high, temp_high, self._max_position_limit, self._max_position_limit, TIME_HIGH, self._max_position_limit]
        if self._pass_scale_interval:
            observation_low.extend([MIN_VOLTAGE, MIN_VOLTAGE])
            observation_high.extend([self._max_voltage, self._max_voltage])
        observation_low = np.array(observation_low)
        observation_high = np.array(observation_high)
        return spaces.Box(low=observation_low, high=observation_high, dtype=np.float)

    def _get_goal_position(self):
        if self._goal_type == 'static':
            return self._goal_start_position
        else:
            return random.uniform(self._goal_min, self._goal_max)

    def _setup_reward_function(self):
        self._reward_gaussian = stats.norm(loc=self._goal_position, scale=self._reward_std)
        self._reward_gaussian_max = self._reward_gaussian.pdf(self._goal_position)

    def is_in_goal(self, position):
        return abs(position - self._goal_position) < self._goal_tolerance

    def create_connection(self, host, port):
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.connect((host, port))
        return connection

    def get_state(self):
        return self._state

    def receive_observation(self):
        global reader
        values = []
        for _ in range(self._values_per_observation):
            value = reader.recv(self._bytes_per_value)
            decoded_value = value.decode('ascii')
            float_value = float(decoded_value)
            values.append(float_value)
        return np.mean(values)

    def receive_observations(self):
        temperature = self.receive_observation()
        position = self.receive_observation()
        return temperature, position

    def fetch_state(self):
        temperature, position = self.receive_observations()
        if self._state is None:
            temperature_change = 0.0
            position_change = 0.0
            time_since_last_step = 0.0
        else:
            temperature_change = temperature - self._state[0]
            position_change = position - self._state[2]
            time_since_last_step = time.time() - self._state_timestep
        state = [temperature, temperature_change, position, position_change, time_since_last_step, self._goal_position]
        if self._pass_scale_interval:
            min_voltage, max_voltage = self.get_action_interval(temperature, position)
            state.extend([min_voltage, max_voltage])
        return np.array(state, dtype=np.float)

    def is_terminal_state(self, state, state_time):
        terminal = self._enter_goal_time is not None and abs(self._enter_goal_time - state_time) >= self._goal_time_tolerance_s
        return terminal

    def reset(self):
        self._enter_goal_time = None
        self._goal_position = self._get_goal_position()
        self._setup_reward_function()
        self._state = None
        self._state_timestep = None
        # wait for the spring to be close to the start position before starting a new trajectory
        while True:
            state = self.fetch_state()
            position = state[2]
            if position <= self._reset_tolerance:
                break
        self._state = state
        self._state_timestep = time.time()
        return state

    def render(self):
        pass

    def reward(self, state, action, next_state):
        position = state[2]
        next_position = next_state[2]
        next_time = next_state[4]
        distance = abs(position - self._goal_position)
        distance_next = abs(next_position - self._goal_position)
        velocity = (distance - distance_next) / next_time
        similarity = self._reward_goal_similarity(next_state)
        return velocity, similarity, velocity + similarity

    def _reward_goal_similarity(self, state):
        position = state[2]
        similarity = self._reward_gaussian.pdf(position)
        similarity_truncated = (self._reward_trunc_min < position) * (position < self._reward_trunc_max) * similarity
        similarity_normalized = similarity_truncated / self._reward_gaussian_max
        return similarity_normalized

    def send_action(self, action):
        global writer
        action_length = self._action_decimal_precision + self._action_digit_precision + 1
        action_string = f'{action:0{action_length}.{self._action_decimal_precision}f}'
        action_encoded = action_string.encode('ascii')
        action_bytes = bytes(action_encoded)
        writer.send(action_bytes)

    def step(self, action):
        """
        Finds the next state in the simulated environmet.

        :param action: action performed in current environment
        :return: (next state, reward)
        """
        if self._scale_action:
            action = self.get_scaled_action(self._state, action)
        self.send_action(action[0])
        # it may be necessary to wait a while in order for the action
        # to have an observable effect
        if self._next_state_wait_time is not None:
            time.sleep(self._next_state_wait_time)
        next_state = self.fetch_state()
        next_state_time = time.time()
        next_position = next_state[1]
        if self._enter_goal_time is None and self.is_in_goal(next_position):
            self._enter_goal_time = next_state_time
        elif self._enter_goal_time is not None and not self.is_in_goal(next_position):
            self._enter_goal_time = None
        velocity, similarity, reward = self.reward(self._state, action, next_state)
        terminal = self.is_terminal_state(next_state, next_state_time)
        info = {'action_scaled': action[0], 'velocity': velocity, 'similarity': similarity}
        self._state = next_state
        self._state_timestep = next_state_time
        return next_state, reward, terminal, info

    def get_scaled_action(self, state, action):
        temperature = state[1]
        position = state[2]
        min, max = self.get_action_interval(temperature, position)
        # transform to interval (0, 1)
        action_normalized = (action - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW)
        # transform to interval (min, max)
        action_scaled = min + (max - min) * action_normalized
        return action_scaled

    def get_action_interval(self, temperature, position):
        """
        Compute min and max voltage.

        Minimum should always be at 0.
        Maximum is linear interpolated as a function of the position until it reaches a configurable threshold.
        It interpolates exponentially from the threshold and until the maximum position.
        Maximum voltage is 0 in the (theoretical) case that position is greater than maximum position.
        """
        if self._max_temperature <= temperature:
            max_voltage = 0.0
        else:
            max_voltage_at_threshold = self._line_slope * position + self._line_intersection
            if position <= self._max_linear_threshold_position:
                max_voltage = max_voltage_at_threshold
            elif position >= self._max_position:
                max_voltage = 0.0
            else:
                distance_to_max_position = abs(position - self._max_position)
                distance_relative_to_max_threshold_position = exp(distance_to_max_position - self._max_position_threshold_distance)
                max_voltage = max_voltage_at_threshold * distance_relative_to_max_threshold_position
        return MIN_VOLTAGE, max_voltage
