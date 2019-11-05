class Environment:

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
        pass

    def step(self, action):
        """
        Step function. Should be implemented by subclass.
        """
        raise NotImplementedError
