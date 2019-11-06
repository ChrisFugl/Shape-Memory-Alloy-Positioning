class Environment:
    """
    Base class that defines an environment. An environment should inherit from this class.
    """

    def __init__(self, options):
        self.options = options
        self.state = self.get_initial_state(options)

    def get_initial_state(self, options):
        raise NotImplementedError('funtion to get initial state is not implemented yet')

    def get_next_state(self, action):
        raise NotImplementedError('function to compute next state is not implemented yet')

    def get_state(self):
        """
        :return: state at current timestep
        """
        return self.state

    def reward(self, state, action, next_state):
        """
        Reward function.

        :param state: current state
        :param action: action performed by the policy
        :param next_state: next state observed after applying action in current state
        :return: reward
        """
        # TODO
        pass

    def step(self, action):
        """
        Finds the next state in the simulated environmet.

        :param action: action performed in current environment
        :return: (next state, reward)
        """
        next_state = self.get_next_state(action)
        reward = self.reward(self.state, ation, next_state)
        self.state = next_state
        return next_state, reward
