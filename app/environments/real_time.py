from app.environments.environment import Environment

class RealTimeEnvironment(Environment):
    """
    Defines a real-time environment.

    state consists of:
        * temperature
        * position

    action consists of:
        * voltage change
    """

    def __init__(self, config):
        """
        :type config: app.config.environments.RealTimeEnvironmentConfig
        """
        self.config = config
        super(RealTimeEnvironment, self).__init__(config)

    def get_initial_state(self, config):
        # TODO
        return None

    def get_next_state(self, action):
        # TODO
        return None

    def is_terminal_state(self, state):
        # TODO
        return False

    def reset(self):
        # TODO
        pass

    def reward(self, state, action, next_state):
        # TODO
        return 0
