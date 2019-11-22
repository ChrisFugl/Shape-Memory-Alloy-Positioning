from app.network import Network
from app.policies.policy import Policy
import torch
from torch import nn
from torch.distributions import Beta

EPSILON = 0.1

class RangePolicy(Policy):
    """
    Predicts an action in a predefined range.
    """

    def __init__(self, config, environment, deterministic=False):
        """
        :type config: app.config.policies.RangePolicyConfig
        :type environment: app.config.environments.EnvironmentConfig
        """
        super(Policy, self).__init__()
        input_size = environment.observation_size
        output_size = 2
        hidden_size = config.network.hidden_size
        number_of_hidden_layers = config.network.number_of_hidden_layers
        self.max = config.max
        self.min = config.min
        self.max_min_difference = config.max - config.min
        self.network = Network(input_size, hidden_size, output_size, number_of_hidden_layers, nn.ReLU())

    def forward(self, observation):
        """
        Forward pass.
        Assumes input is a torch tensor.

        :type observation: torch.Tensor
        """
        return self.generate_action(observation)

    def get_action(self, observation):
        """
        Computes action from the given observation (state).
        Assumes input is a numpy array.
        """
        observation_torch = torch.from_numpy(observation).float()
        action_torch, _ = self.generate_action(observation_torch)
        action_numpy = action_torch.detach().numpy()
        return action_numpy

    def generate_action(self, observation):
        """
        Computes action from observation (state).
        Assumes input is a torch tensor.
        """
        # add epsilon to ensure that output is greater than zero
        network_output = self.network(observation) + EPSILON
        alpha = network_output[:, 0]
        beta = network_output[:, 1]
        distribution = Beta(alpha, beta)
        sample = distribution.rsample()
        # transform to range (min, max)
        action = self.min + self.max_min_difference * sample
        action.requires_grad_()
        log_probability = distribution.log_prob(sample)
        action = action.unsqueeze(1)
        log_probability = log_probability.unsqueeze(1)
        return action, log_probability
