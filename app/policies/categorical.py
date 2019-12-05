from app.network import Network
from app.policies.policy import Policy
import torch
from torch.distributions import Categorical

class CategoricalPolicy(Policy):
    """
    Policy with a discrete action set.
    """

    name = 'categorical'

    def __init__(self, config, environment):
        """
        :type config: app.config.policies.CategoricalPolicyConfig
        :type environment: app.config.environments.EnvironmentConfig
        """
        super(Policy, self).__init__()
        input_size = environment.observation_size
        output_size = len(config.actions)
        hidden_size = config.network.hidden_size
        number_of_hidden_layers = config.network.number_of_hidden_layers
        self.actions = torch.tensor(config.actions)
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
        logits = self.network(observation)
        categorical = Categorical(logits=logits)
        sample = categorical.sample()
        action = self.actions[sample]
        action.requires_grad_()
        log_probability = categorical.log_prob(sample)
        action = action.unsqueeze(1)
        log_probability = log_probability.unsqueeze(1)
        return action, log_probability
