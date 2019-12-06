from app.config.environments import DebugEnvironmentConfig, RealTimeEnvironmentConfig, SimulatedEnvironmentConfig, TestEnvironmentConfig
from app.config.policies import CategoricalPolicyConfig, GaussianPolicyConfig, RangePolicyConfig, TanhGaussianPolicyConfig, TestPolicyConfig
from app.config.model import ModelConfig

_environment_classes = {
    'debug': DebugEnvironmentConfig,
    'real_time': RealTimeEnvironmentConfig,
    'simulated': SimulatedEnvironmentConfig,
    'test': TestEnvironmentConfig
}

_policy_classes = {
    'categorical': CategoricalPolicyConfig,
    'gaussian': GaussianPolicyConfig,
    'range': RangePolicyConfig,
    'tanh_gaussian': TanhGaussianPolicyConfig,
    'test': TestPolicyConfig
}

class Config:

    def __init__(self, *,
        batch_size=128,
        collect_actions,
        collect_actions_every,
        environment,
        exploration_steps=256,
        evaluation_steps=256,
        final_position=0.05,
        gradient_steps=1,
        iterations=100000,
        max_buffer_size=100000,
        max_trajectory_length=30,
        min_num_steps_before_training=0,
        model,
        policy,
        save_model=None
    ):
        environment_type = environment.pop('type')
        environment_class = _environment_classes[environment_type]
        policy_type = policy.pop('type')
        policy_class = _policy_classes[policy_type]
        self.batch_size = batch_size
        self.collect_actions = collect_actions
        self.collect_actions_every = collect_actions_every
        self.environment = environment_class(**environment)
        self.exploration_steps = exploration_steps
        self.evaluation_steps = evaluation_steps
        self.final_position = final_position
        self.gradient_steps = gradient_steps
        self.iterations = iterations
        self.max_buffer_size = max_buffer_size
        self.max_trajectory_length = max_trajectory_length
        self.min_num_steps_before_training = min_num_steps_before_training
        self.model = ModelConfig(**model)
        self.policy = policy_class(**policy)
        self.save_model = save_model

        self.environment_type = environment_type
        self.policy_type = policy_type

    def __str__(self):
        return (f'Config(\n'
             + f'  collect_actions = {self.collect_actions}\n'
             + f'  collect_actions_every = {self.collect_actions_every}\n'
             + f'  environment = {self.environment}\n'
             + f'  exploration_steps = {self.exploration_steps}\n'
             + f'  evaluation_steps = {self.evaluation_steps}\n'
             + f'  final_position = {self.final_position}\n'
             + f'  gradient_steps = {self.gradient_steps}\n'
             + f'  iterations = {self.iterations}\n'
             + f'  max_buffer_size = {self.max_buffer_size}\n'
             + f'  max_trajectory_length = {self.max_trajectory_length}\n'
             + f'  min_num_steps_before_training = {self.min_num_steps_before_training}\n'
             + f'  model = {self.model}\n'
             + f'  policy = {self.policy}\n'
             + f'  save_model = {self.save_model}\n'
             + ')')
