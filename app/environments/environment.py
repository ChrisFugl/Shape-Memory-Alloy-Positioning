from math import exp
from scipy.spatial import distance

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

    def is_terminal_state(self):
        """
        :return: boolean whether current state is the terminal state
        """
        raise NotImplementedError('function to compute if the current state is a terminal state is not implemented yet')

    def reward(self, state, action, next_state):
        """
        Reward function.

        :param state: current state
        :param action: action performed by the policy
        :param next_state: next state observed after applying action in current state
        :return: reward
        """
        # TODO: include speed towards goal position?
        _, _, position_next = next_state
        position_goal_distance = distance.euclidean([position_next], [self.options.final_position])
        position_goal_similarity = exp(- position_goal_distance)
        return position_goal_similarity

    def step(self, action):
        """
        Finds the next state in the simulated environmet.

        :param action: action performed in current environment
        :return: (next state, reward)
        """
        next_state = self.get_next_state(action)
        reward = self.reward(self.state, action, next_state)
        terminal = self.is_terminal_state()
        self.state = next_state
        return next_state, reward, terminal
