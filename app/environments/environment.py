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
        self.state = self.get_initial_state()

    def get_initial_state(self):
        raise NotImplementedError('funtion to get initial state is not implemented yet')

    def get_next_state(self, action):
        raise NotImplementedError('function to compute next state is not implemented yet')

    def get_state(self):
        """
        :return: state at current timestep
        """
        return self.state

    def is_terminal_state(self, state):
        """
        :return: boolean whether current state is the terminal state
        """
        raise NotImplementedError('function to compute if the current state is a terminal state is not implemented yet')

    def reset(self):
        raise NotImplementedError('reset method should be implemented by subclass')

    def reward(self, state, action, next_state):
        """
        :return: the reward from taking an action in a given state
        """
        raise NotImplementedError('the reward function should be implemented by the subclass')

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
