from app.environments.environment import Environment
from gym import spaces
from math import cos, exp, pi
import numpy as np
from scipy.spatial import distance

ACTION_LOW = -5.0
ACTION_HIGH = 5.0
DEFLECTION_LOW = 0.0
DEFLECTION_HIGH = 100.0
TEMP_LOW = 0.0
TEMP_HIGH = 100.0
FORCE_LOW = 0.0
FORCE_HIGH = 10.0

class SimulatedEnvironment(Environment):
    """
    Defines a simulated environment.

    state consists of:
        * temperature
        * temperature change
        * deflection

    action consists of:
        * temperature change
    note that temperature is used instead of voltage
    since we already have equations to compute deflection,
    but we do not currently have equations to compute
    temperature (and so deflection) from voltage
    """

    name = 'simulated'

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

        self.action_space = spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, shape=(config.action_size,), dtype=np.float)
        self.observation_space = spaces.Box(
            low=np.array([TEMP_LOW, ACTION_LOW, DEFLECTION_LOW]),
            high=np.array([TEMP_HIGH, ACTION_HIGH, DEFLECTION_HIGH]),
            dtype=np.float)

    def get_initial_state(self):
        return np.array([
            self.config.initial_temperature, # temperature
            0, # temperature change
            self.config.initial_deflection, # deflection
            # self.config.initial_position # position (meter),
        ], dtype=np.float)

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

    def is_in_goal(self, position):
        return abs(position - self.config.final_position) < self.config.goal_tolerance

    def is_terminal_state(self, state):
        return self.is_in_goal(state[2])

    def render(self):
        pass

    def reset(self):
        self.state = self.get_initial_state()
        return self.state

    def reward(self, state, action, next_state):
        """
        Reward function.

        :param state: current state
        :param action: action performed by the policy
        :param next_state: next state observed after applying action in current state
        :return: reward
        """
        position = state[2]
        next_position = next_state[2]
        distance_to_goal = abs(position - self.config.final_position)
        next_distance_to_goal = abs(next_position- self.config.final_position)
        distance_difference = distance_to_goal - next_distance_to_goal
        if self.is_in_goal(next_position):
            next_similarity_to_goal = 1
        else:
            next_similarity_to_min_goal = exp(-abs(next_position - (self.config.final_position - self.config.goal_tolerance)))
            next_similarity_to_max_goal = exp(-abs(next_position - (self.config.final_position + self.config.goal_tolerance)))
            next_similarity_to_goal = max(next_similarity_to_min_goal, next_similarity_to_max_goal)
        return distance_difference + next_similarity_to_goal

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
        return next_state, reward, terminal, {}
