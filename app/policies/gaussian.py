from app.network import Network
from app.policies.policy import Policy
import torch
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class GaussianPolicy(Policy):
    """
    Poliy which makes use of a gaussian to predict a continuous action.
    """

    name = 'gaussian'

    def __init__(self, config, environment, deterministic=False):
        """
        :type config: app.config.policies.GaussianPolicyConfig
        :type environment: app.config.environments.EnvironmentConfig
        """
        super(Policy, self).__init__()
        input_size = environment.observation_size
        output_size = 2 # mean and log(std)
        hidden_size = config.network.hidden_size
        number_of_hidden_layers = config.network.number_of_hidden_layers
        self.deterministic = deterministic
        self.network = Network(input_size, hidden_size, output_size, number_of_hidden_layers)

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
        network_output = self.network(observation)
        mean = network_output[:, 0]
        log_std = torch.clamp(network_output[:, 1], LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        mean_zeros = torch.zeros(mean.size())
        std_ones = torch.ones(std.size())
        gaussian = Normal(mean_zeros, std_ones)

        if self.deterministic:
            action = mean
        else:
            sample = gaussian.sample()
            action = mean + std * sample
            action.requires_grad_()

        log_probability = gaussian.log_prob(action)

        action = action.unsqueeze(1)
        log_probability = log_probability.unsqueeze(1)
        return action, log_probability
