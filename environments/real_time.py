from environments.environment import Environment

class RealTimeEnvironment(Environment):

    def __init__(self, options):
        self.state = self.get_initial_state(options)

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

    def get_initial_state(self, options):
        # TODO
        return None

    def get_next_state(self, action):
        # TODO
        return None
