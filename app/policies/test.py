from app.policies.policy import Policy

class TestPolicy(Policy):
    """
    Policy that should only be used for testing purposes.
    """

    def __init__(self, config, envionment):
        """
        :type config: app.config.policies.TestPolicyConfig
        :type environment: app.config.environments.EnvironmentConfig
        """
        super(TestPolicy, self).__init__()
        self.change = config.change

    def set_change(self, change):
        self.change = change

    def get_action(self, state):
        return [self.change]
