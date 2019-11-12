from environments.environment import Environment
from math import cos, pi
import numpy as np

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

    def __init__(self, options):
        super(SimulatedEnvironment, self).__init__(options)
        spring_constant = self.options.spring_diameter / self.options.wire_diameter
        wahls_correction = (4 * spring_constant - 1) / (4 * spring_constant - 4) + (0.615 / spring_constant)
        shear_stress_corrected = options.shear_stress * wahls_correction
        c1 = shear_stress_corrected
        c2 = options.wire_diameter / (options.number_of_coils * pi * options.spring_diameter ** 2)
        critical_detwinning_stress_difference = options.critical_detwinning_starting_stress - options.critical_detwinning_finishing_stress
        tmsm = options.twinned_martensite_shear_modulus
        mrd = options.max_recoverable_deflection
        self.initial_martensitic_fraction_of_twinned_martensite = (- (c1 * options.initial_force) / (c2 * tmsm)) / mrd
        self.martensite_cos_ratio = pi / critical_detwinning_stress_difference
        self.austenite_temperature_difference = options.austenitic_finish_temperature - options.austenitic_start_temperature

    def get_initial_state(self, options):
        return [
            self.options.initial_temperature, # temperature
            self.options.initial_deflection, # deflection
            self.options.initial_position # position (meter)
        ]

    def get_next_state(self, action):
        """
        Estimates next state in simulated environment.

        :param action: action as 1x1 numpy.ndarray containing the temperature change
        :return: next state
        """
        temperature, _, position = self.state

        # find next temperature
        temperature_change = action[0]
        temperature_next = temperature + temperature_change

        # find next displacement
        if temperature_change < 0:
            sigma = self.get_cooling_sigma(temperature_next)
        else:
            sigma = self.get_heating_sigma(temperature_next)
        # it is a bad practice to multiply by an unanmed constant (here 55)
        # we should assign this to a variable and give it a good descriptive name
        displacement_next = sigma * self.options.max_recoverable_deflection * 55

        # find next position
        # TODO: is this correct?
        if temperature_change < 0:
            position_next = position + (position * displacement_next)
        else:
            position_next = position - (position * displacement_next)

        return np.array([temperature_next, displacement_next, position_next], dtype=np.float)

    def get_cooling_sigma(self, temperature_next):
        if temperature_next < self.options.martensitic_finish_temperature:
            return 1
        elif temperature_next > self.options.martensitic_start_temperature:
            return 0
        else:
            martensite_temperature_difference = temperature_next - self.options.martensitic_start_temperature
            cos_multiplier = self.options.shear_stress - self.options.critical_detwinning_finishing_stress
            cos_multiplier = cos_multiplier - self.options.martensitic_constant * martensite_temperature_difference
            sigma = 1 - self.initial_martensitic_fraction_of_twinned_martensite
            sigma = sigma / 2
            sigma = sigma * cos(self.martensite_cos_ratio * cos_multiplier)
            sigma = sigma + (1 + self.initial_martensitic_fraction_of_twinned_martensite) / 2
            return sigma

    def get_heating_sigma(self, temperature_next):
        temperature_next_austenic_difference = temperature_next - self.options.austenitic_start_temperature
        if temperature_next_austenic_difference < 0:
            return 1
        else:
            multiplicand = self.options.sigma_o / 2
            multiplier = cos(pi / self.austenite_temperature_difference * temperature_next_austenic_difference)
            multiplier = multiplier + 1
            return multiplicand * multiplier
