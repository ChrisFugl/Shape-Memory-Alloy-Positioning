from torch import nn

class Policy(nn.Module):
    """
    Base policy that defines the interface that all policies must use.
    """

    def __init__(self):
        super(Policy, self).__init__()

    def get_action(self, state):
        """
        Computes the action given the current state.
        """
        raise NotImplementedError('get_action must be implemented by the subclass')
