class Environment:
    """
    Base class that defines an environment. An environment should inherit from this class.
    """

    def __init__(self, config):
        """
        :type config: app.config.environments.EnvironmentConfig
        """
        self.action_size = config.action_size
        self.observation_size = config.observation_size

    def get_state(self):
        """
        :return: state at current timestep
        """
        raise NotImplementedError('get_state should be implemented by a subclass')

    def is_terminal_state(self, state):
        """
        :return: boolean whether current state is the terminal state
        """
        raise NotImplementedError('is_terminal_state should be implemented by a subclass')

    def reset(self):
        raise NotImplementedError('reset method should be implemented by subclass')

    def reward(self, state, action, next_state):
        """
        :return: the reward from taking an action in a given state
        """
        raise NotImplementedError('reward should be implemented by a subclass')

    def step(self, action):
        """
        Finds the next state in the simulated environmet.

        :param action: action performed in current environment
        :return: (next state, reward)
        """
        raise NotImplementedError('step should be implemented by a subclass')
