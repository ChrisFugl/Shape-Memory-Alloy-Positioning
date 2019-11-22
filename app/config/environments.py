class EnvironmentConfig:

    def __init__(self, observation_size, action_size):
        """
        :type observation_size: int
        """
        self.action_size = action_size
        self.observation_size = observation_size

class DebugEnvironmentConfig(EnvironmentConfig):

    def __init__(self, *, min_start_position=0, max_start_position=100, goal_position=50):
        action_size = 1
        observation_size = 1
        super(DebugEnvironmentConfig, self).__init__(observation_size=observation_size, action_size=action_size)
        self.min_start_position = min_start_position
        self.max_start_position = max_start_position
        self.goal_position = goal_position

class RealTimeEnvironmentConfig(EnvironmentConfig):

    def __init__(self, *,
        action_decimal_precision,
        action_digit_precision,
        bytes_per_value,
        host,
        port_read,
        port_write,
        values_per_observation
    ):
        action_size = 1
        observation_size = 3
        super(RealTimeEnvironmentConfig, self).__init__(observation_size=observation_size, action_size=action_size)
        self.action_decimal_precision = action_decimal_precision
        self.action_digit_precision = action_digit_precision
        self.bytes_per_value = bytes_per_value
        self.host = host
        self.port_read = port_read
        self.port_write = port_write
        self.values_per_observation = values_per_observation

class SimulatedEnvironmentConfig(EnvironmentConfig):

    def __init__(self, *,
        final_position=0.05,
        initial_deflection=0,
        initial_force=0,
        initial_martensitic_fraction_of_detwinned_martensite=1,
        initial_position=0.12,
        initial_temperature=20,
        number_of_coils=55,
        spring_diameter=0.0025,
        wire_diameter=0.0005,
        twinned_martensite_shear_modulus=50000000000,
        austenite_shear_modulus=100000000000,
        austenitic_start_temperature=59,
        austenitic_finish_temperature=71,
        austenitic_constant=100000000,
        martensitic_start_temperature=55,
        martensitic_finish_temperature=43,
        martensitic_constant=40000000,
        max_recoverable_deflection=0.001,
        critical_detwinning_starting_stress=500000000.0,
        critical_detwinning_finishing_stress=1020000000.0,
        delta_max=1.09,
        shear_stress=1600000000.0,
        force_applied=4,
        sigma_o=1
    ):
        action_size = 1
        observation_size = 3
        super(SimulatedEnvironmentConfig, self).__init__(observation_size=observation_size, action_size=action_size)
        self.austenitic_finish_temperature = austenitic_finish_temperature
        self.austenite_shear_modulus = austenite_shear_modulus
        self.austenitic_start_temperature = austenitic_start_temperature
        self.austenitic_constant = austenitic_constant
        self.critical_detwinning_finishing_stress = critical_detwinning_finishing_stress
        self.critical_detwinning_starting_stress = critical_detwinning_starting_stress
        self.delta_max = delta_max
        self.final_position = final_position
        self.force_applied = force_applied
        self.initial_deflection = initial_deflection
        self.initial_force = initial_force
        self.initial_martensitic_fraction_of_detwinned_martensite = initial_martensitic_fraction_of_detwinned_martensite
        self.initial_position = initial_position
        self.initial_temperature = initial_temperature
        self.martensitic_finish_temperature = martensitic_finish_temperature
        self.martensitic_start_temperature = martensitic_start_temperature
        self.martensitic_constant = martensitic_constant
        self.max_recoverable_deflection = max_recoverable_deflection
        self.number_of_coils = number_of_coils
        self.shear_stress = shear_stress
        self.sigma_o = sigma_o
        self.spring_diameter = spring_diameter
        self.twinned_martensite_shear_modulus = twinned_martensite_shear_modulus
        self.wire_diameter = wire_diameter

    def __str__(self):
        return (f'SimulatedEnvironment(\n'
             + f'    austenitic_finish_temperature = {self.austenitic_finish_temperature}\n'
             + f'    austenite_shear_modulus = {self.austenite_shear_modulus}\n'
             + f'    austenitic_start_temperature = {self.austenitic_start_temperature}\n'
             + f'    austenitic_constant = {self.austenitic_constant}\n'
             + f'    critical_detwinning_finishing_stress = {self.critical_detwinning_finishing_stress}\n'
             + f'    critical_detwinning_starting_stress = {self.critical_detwinning_starting_stress}\n'
             + f'    delta_max = {self.delta_max}\n'
             + f'    force_applied = {self.force_applied}\n'
             + f'    initial_deflection = {self.initial_deflection}\n'
             + f'    initial_force = {self.initial_force}\n'
             + f'    initial_martensitic_fraction_of_detwinned_martensite = {self.initial_martensitic_fraction_of_detwinned_martensite}\n'
             + f'    initial_position = {self.initial_position}\n'
             + f'    initial_temperature = {self.initial_temperature}\n'
             + f'    martensitic_finish_temperature = {self.martensitic_finish_temperature}\n'
             + f'    martensitic_start_temperature = {self.martensitic_start_temperature}\n'
             + f'    max_recoverable_deflection = {self.max_recoverable_deflection}\n'
             + f'    number_of_coils = {self.number_of_coils}\n'
             + f'    shear_stress = {self.shear_stress}\n'
             + f'    sigma_o = {self.sigma_o}\n'
             + f'    spring_diameter = {self.spring_diameter}\n'
             + f'    twinned_martensite_shear_modulus = {self.twinned_martensite_shear_modulus}\n'
             + f'    wire_diameter = {self.wire_diameter}\n'
             + '  )')

class TestEnvironmentConfig(EnvironmentConfig):

    def __init__(self, *args, initial_state=0, final_state=2):
        action_size = 1
        observation_size = 1
        super(TestEnvironmentConfig, self).__init__(observation_size=observation_size, action_size=action_size)
        self.initial_state = initial_state
        self.final_state = final_state

    def __str__(self):
        return (f'TestEnvironment(\n'
             + f'    initial_state = {self.initial_state}\n'
             + f'    final_state = {self.final_state}\n'
             + '  )')
