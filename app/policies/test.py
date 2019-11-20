from app.policies.policy import Policy

class TestPolicy(Policy):
    """
    Policy that should only be used for testing purposes.
    """

    def __init__(self, change=0):
        super(TestPolicy, self).__init__()
        self.change = change

    def set_change(self, change):
        self.change = change

    def get_action(self, state):
        return [self.change]
