from app.config.environments import DebugEnvironmentConfig, RealTimeEnvironmentConfig, SimulatedEnvironmentConfig, TestEnvironmentConfig
from app.config.policies import GaussianPolicyConfig, TestPolicyConfig
from app.config.model import ModelConfig

_environment_classes = {
    'debug': DebugEnvironmentConfig,
    'real_time': RealTimeEnvironmentConfig,
    'simulated': SimulatedEnvironmentConfig,
    'test': TestEnvironmentConfig
}

_policy_classes = {
    'gaussian': GaussianPolicyConfig,
    'test': TestPolicyConfig
}

class Config:

    def __init__(self, *,
        batch_size=128,
        environment,
        environment_steps=10,
        final_position=0.05,
        gradient_steps=1,
        iterations=100000,
        max_buffer_size=100000,
        max_trajectory_length=30,
        model,
        policy,
        save_model=None
    ):
        environment_type = environment.pop('type')
        environment_class = _environment_classes[environment_type]
        policy_type = policy.pop('type')
        policy_class = _policy_classes[policy_type]
        self.batch_size = batch_size
        self.environment = environment_class(**environment)
        self.environment_steps = environment_steps
        self.final_position = final_position
        self.gradient_steps = gradient_steps
        self.iterations = iterations
        self.max_buffer_size = max_buffer_size
        self.max_trajectory_length = max_trajectory_length
        self.model = ModelConfig(**model)
        self.policy = policy_class(**policy)
        self.save_model = save_model

        self.environment_type = environment_type
        self.policy_type = policy_type

    def __str__(self):
        return (f'Config(\n'
             + f'  environment = {self.environment}\n'
             + f'  final_position = {self.final_position}\n'
             + f'  model = {self.model}\n'
             + f'  policy = {self.policy}\n'
             + f'  save_model = {self.save_model}\n'
             + ')')
