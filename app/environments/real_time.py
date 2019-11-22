from app.environments.environment import Environment
import numpy as np
import socket
import struct
import time

class RealTimeEnvironment(Environment):
    """
    Real-time environment.

    state:
        * temperature
        * position
        * force

    action:
        * voltage
    """

    def __init__(self, config):
        """
        :type config: app.config.environments.RealTimeEnvironmentConfig
        """
        super(RealTimeEnvironment, self).__init__(config)
        self.action_decimal_precision = config.action_decimal_precision
        self.action_digit_precision = config.action_digit_precision
        self.bytes_per_value = config.bytes_per_value
        self.values_per_observation = config.values_per_observation

        if config.port_read == config.port_write:
            client = self.create_connection(config.host, config.port_read)
            self.reader = client
            self.writer = client
        else:
            self.reader = self.create_connection(config.host, config.port_read)
            self.writer = self.create_connection(config.host, config.port_write)

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
        temperature = self.receive_observation()
        position = self.receive_observation()
        force = self.receive_observation()
        return temperature, position, force

    def fetch_state(self):
        observations = self.receive_observations()
        print(f'received: {observations}')
        return list(observations)

    def is_terminal_state(self, state):
        # TODO
        return False

    def reset(self):
        self.state = self.fetch_state()

    def reward(self, state, action, next_state):
        # TODO
        return 0

    def send_action(self, action):
        action_length = self.action_decimal_precision + self.action_digit_precision + 1
        action_string = f'{action:0{action_length}.{self.action_decimal_precision}f}'
        print(f'sending: {action_string}')
        action_encoded = action_string.encode('ascii')
        action_bytes = bytes()
        self.writer.send(action_bytes)

    def step(self, action):
        """
        Finds the next state in the simulated environmet.

        :param action: action performed in current environment
        :return: (next state, reward)
        """
        time.sleep(0.5)
        self.send_action(action[0])
        time.sleep(0.5)
        next_state = self.fetch_state()
        reward = self.reward(self.state, action, next_state)
        terminal = self.is_terminal_state(next_state)
        self.state = next_state
        return next_state, reward, terminal
