from app.config.network import NetworkConfig

class PolicyConfig:
    pass

class CategoricalPolicyConfig(PolicyConfig):

    def __init__(self, *, actions, network):
        self.actions = actions
        self.network = NetworkConfig(**network)

    def __str__(self):
        return (f'CategoricalPolicy(\n'
             + f'    actions = {self.actions}\n'
             + f'    network = {self.network}\n'
             + '  )')

class GaussianPolicyConfig(PolicyConfig):

    def __init__(self, *, network):
        self.network = NetworkConfig(**network)

    def __str__(self):
        return (f'GaussianPolicy(\n'
             + f'    network = {self.network}\n'
             + '  )')

class RangePolicyConfig(PolicyConfig):

    def __init__(self, *, max, min, network):
        self.max = max
        self.min = min
        self.network = NetworkConfig(**network)

    def __str__(self):
        return (f'RangePolicyConfig(\n'
             + f'    max = {self.max}\n'
             + f'    min = {self.min}\n'
             + f'    network = {self.network}\n'
             + '  )')

class TestPolicyConfig(PolicyConfig):

    def __init__(self, *, change):
        self.change = change

    def __str__(self):
        return (f'TestPolicy(\n'
             + f'    change = {self.change}\n'
             + '  )')
