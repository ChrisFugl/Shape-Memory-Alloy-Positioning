from app.config.network import NetworkConfig

class PolicyConfig:
    pass

class GaussianPolicyConfig(PolicyConfig):

    def __init__(self, *, network):
        self.network = NetworkConfig(**network)

    def __str__(self):
        return (f'GaussianPolicy(\n'
             + f'    network = {self.network}\n'
             + '  )')

class TestPolicyConfig(PolicyConfig):

    def __init__(self, *, change):
        self.change = change

    def __str__(self):
        return (f'TestPolicy(\n'
             + f'    change = {self.change}\n'
             + '  )')
