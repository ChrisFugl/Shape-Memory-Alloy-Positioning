from app.network import Network
from app.policies.policy import Policy
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import eval_np
from rlkit.torch.networks import Mlp
import torch
from torch import nn
from torch.distributions import Beta

EPSILON = 10e-9

class RangePolicy(Mlp, ExplorationPolicy):
    """
    Predicts an action in a predefined range.
    """

    name = 'range'

    def __init__(self, config, environment):
        """
        :type config: app.config.policies.RangePolicyConfig
        :type environment: app.config.environments.EnvironmentConfig
        """
        hidden_sizes = [config.network.hidden_size] * config.network.number_of_hidden_layers
        input_size = environment.observation_size
        output_size = 2
        init_w = 1e-3
        super().__init__(hidden_sizes, input_size=input_size, output_size=output_size, init_w=init_w)
        self.output_activation = nn.ReLU()
        self.min = config.min
        self.max_min_difference = config.max - config.min
        self.max_min_difference_squared = self.max_min_difference * self.max_min_difference

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(self, observation, reparameterize=True, deterministic=False, return_log_prob=False):
        """
        Forward pass.
        Assumes input is a torch tensor.

        :type observation: torch.Tensor
        """
        layer_input = observation
        for fc in self.fcs:
            layer_input = self.hidden_activation(fc(layer_input))
        network_output = self.output_activation(self.last_fc(layer_input))

        alpha = network_output[:, 0].unsqueeze(1) + EPSILON
        beta = network_output[:, 1].unsqueeze(1) + EPSILON
        distribution = Beta(alpha, beta)
        distribution_mean = distribution.mean
        if deterministic:
            sample = distribution.rsample()
        else:
            sample = distribution_mean
        print(sample.min(), sample.max())
        # transform to range (min, max)
        action = self.min + self.max_min_difference * sample
        mean = self.min + self.max_min_difference * distribution_mean
        variance = self.max_min_difference_squared * distribution.variance
        std = torch.sqrt(variance)
        log_std = torch.log(std)
        log_prob = distribution.log_prob(sample)
        entropy = distribution.entropy()
        mean_action_log_prob = None
        pre_tanh_value = None
        return action, mean, log_std, log_prob, entropy, std, mean_action_log_prob, pre_tanh_value
