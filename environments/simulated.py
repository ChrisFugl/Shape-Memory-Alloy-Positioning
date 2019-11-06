from environments.environment import Environment

class SimulatedEnvironment(Environment):
    """
    Defines a simulated environment.

    state consists of:
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
        spring_constant = self.options.initial_spring_diameter / self.options.wire_diameter
        wahls_correction = (4 * spring_constant - 1) / (4 * spring_constant - 4) + (0.615 / spring_constant)


    def get_initial_state(self, options):
        # TODO
        return [
            0, # deflection
            0.12 # position (meter)
        ]

    def get_next_state(self, action):
        """
        Estimates next state in simulated environment.

        :param action: action as 1x1 numpy.ndarray containing the temperature change
        :return: next state
        """
        # F is force (configurable)
        # sigma_S and sigma_T are to be determined in experiments (configurable)
        # sigma_oS is a type, it is sigma_So

        temperature_change = action[0]
