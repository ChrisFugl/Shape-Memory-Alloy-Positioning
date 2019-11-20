class Policy:
    """
    Base policy defines the interface that all policies must use.
    """

    def get_action(self, state):
        """
        Computes the action given the current state.
        """
        raise NotImplementedError('get_action must be implemented by the subclass')
