from app.environments.environment import Environment
from math import cos, exp, pi
import numpy as np
from scipy.spatial import distance

class SimulatedEnvironment(Environment):
    """
    Defines a simulated environment.

    state consists of:
        * temperature
        * deflection
        * position

    action consists of:
        * temperature change
    note that temperature is used instead of voltage
    since we already have equations to compute deflection,
    but we do not currently have equations to compute
    temperature (and so deflection) from voltage
    """

    def __init__(self, config):
        """
        :type config: app.config.environments.SimulatedEnvironmentConfig
        """
        super(SimulatedEnvironment, self).__init__(config)
        spring_constant = config.spring_diameter / config.wire_diameter
        wahls_correction = (4 * spring_constant - 1) / (4 * spring_constant - 4) + (0.615 / spring_constant)
        shear_stress_corrected = config.shear_stress * wahls_correction
        c1 = shear_stress_corrected
        c2 = config.wire_diameter / (config.number_of_coils * pi * config.spring_diameter ** 2)
        critical_detwinning_stress_difference = config.critical_detwinning_starting_stress - config.critical_detwinning_finishing_stress
        tmsm = config.twinned_martensite_shear_modulus
        mrd = config.max_recoverable_deflection
        self.max_recoverable_deflection = mrd
        self.initial_martensitic_fraction_of_twinned_martensite = (- (c1 * config.initial_force) / (c2 * tmsm)) / mrd
        self.martensite_cos_ratio = pi / critical_detwinning_stress_difference
        self.austenite_temperature_difference = config.austenitic_finish_temperature - config.austenitic_start_temperature
        self.config = config
        self.state = self.get_initial_state()

    def get_initial_state(self):
        return [
            self.config.initial_temperature, # temperature
            0, # temperature change
            self.config.initial_deflection, # deflection
            # self.config.initial_position # position (meter),
        ]

    def get_next_state(self, action):
        """
        Estimates next state in the simulated environment.

        :param action: temperature change
        :type action: float
        :return: next state
        """
        temperature, _, _ = self.state

        # find next temperature
        next_temperature_change = action
        temperature_next = temperature + next_temperature_change

        # find next displacement
        if next_temperature_change < 0:
            sigma = self.get_cooling_sigma(temperature_next)
        else:
            sigma = self.get_heating_sigma(temperature_next)
        # it is a bad practice to multiply by an unanmed constant (here 55)
        # we should assign this to a variable and give it a good descriptive name
        displacement_next = sigma * self.config.max_recoverable_deflection * 55

        # find next position
        # position_next = position * displacement_next

        # return np.array([temperature_next, displacement_next, position_next], dtype=np.float)
        return np.array([temperature_next, next_temperature_change, displacement_next], dtype=np.float)

    def get_state(self):
        return self.state

    def get_cooling_sigma(self, temperature_next):
        if temperature_next < self.config.martensitic_finish_temperature:
            return 1
        elif temperature_next > self.config.martensitic_start_temperature:
            return 0
        else:
            martensite_temperature_difference = temperature_next - self.config.martensitic_start_temperature
            cos_multiplier = self.config.shear_stress - self.config.critical_detwinning_finishing_stress
            cos_multiplier = cos_multiplier - self.config.martensitic_constant * martensite_temperature_difference
            sigma = 1 - self.initial_martensitic_fraction_of_twinned_martensite
            sigma = sigma / 2
            sigma = sigma * cos(self.martensite_cos_ratio * cos_multiplier)
            sigma = sigma + (1 + self.initial_martensitic_fraction_of_twinned_martensite) / 2
            return sigma

    def get_heating_sigma(self, temperature_next):
        temperature_next_austenic_difference = temperature_next - self.config.austenitic_start_temperature
        if temperature_next_austenic_difference < 0:
            return 1
        else:
            multiplicand = self.config.sigma_o / 2
            multiplier = cos(pi / self.austenite_temperature_difference * temperature_next_austenic_difference)
            multiplier = multiplier + 1
            return multiplicand * multiplier

    def is_terminal_state(self, state):
        epsilon = 10 ** -8
        return abs(state[2] - self.config.final_position) < epsilon

    def reset(self):
        self.state = self.get_initial_state()

    def reward(self, state, action, next_state):
        """
        Reward function.

        :param state: current state
        :param action: action performed by the policy
        :param next_state: next state observed after applying action in current state
        :return: reward
        """
        # TODO: include speed towards goal position?
        # _, _, displacement = state
        _, _, displacement_next = next_state
        # goal_distance = distance.euclidean([displacement], [self.config.final_position])
        next_goal_distance = distance.euclidean([displacement_next], [self.config.final_position])
        next_goal_similarity = exp(- next_goal_distance)
        # return (next_goal_distance - goal_distance) * next_goal_similarity
        return next_goal_similarity

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
