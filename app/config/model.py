from app.config.network import NetworkConfig

class ModelConfig:

    def __init__(self, *args,
        discount_factor=0.99,
        exponential_weight=0.005,
        learning_rate_policy=0.0001,
        learning_rate_q=0.0001,
        network,
        reward_scale=1.0,
        temperature=0.
    ):
        self.discount_factor = discount_factor
        self.exponential_weight = exponential_weight
        self.learning_rate_policy = learning_rate_policy
        self.learning_rate_q = learning_rate_q
        self.network = NetworkConfig(**network)
        self.reward_scale = reward_scale
        self.temperature = temperature

    def __str__(self):
        return (f'Model(\n'
             + f'    discount_factor = {self.discount_factor}\n'
             + f'    exponential_weight = {self.exponential_weight}\n'
             + f'    learning_rate_policy = {self.learning_rate_policy}\n'
             + f'    learning_rate_q = {self.learning_rate_q}\n'
             + f'    network = {self.network}\n'
             + f'    reward_scale = {self.reward_scale}\n'
             + f'    temperature = {self.temperature}\n'
             + '  )')
