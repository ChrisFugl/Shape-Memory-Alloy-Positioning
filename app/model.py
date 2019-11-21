from app.network import Network
from app.policies import get_policy
import torch
from torch import optim, nn

class Model:
    """
    Soft Actor-Critic model.
    """

    def __init__(self, config, environment, policy):
        """
        :type config: app.config.model.ModelConfig
        :type environment: app.environments.environment.Environment
        :type policy: app.policies.policy.Policy
        """
        q_output_size = 1
        q_input_size = environment.observation_size + environment.action_size
        hidden_size = config.network.hidden_size
        number_of_hidden_layers = config.network.number_of_hidden_layers
        self.policy = policy
        self.reward_scale = config.reward_scale
        self.temperature = config.temperature
        self.discount_factor = config.discount_factor
        self.exponential_weight = config.exponential_weight
        self.q1 = Network(q_input_size, hidden_size, q_output_size, number_of_hidden_layers, nn.ReLU())
        self.q2 = Network(q_input_size, hidden_size, q_output_size, number_of_hidden_layers, nn.ReLU())
        self.target_q1 = Network(q_input_size, hidden_size, q_output_size, number_of_hidden_layers, nn.ReLU())
        self.target_q2 = Network(q_input_size, hidden_size, q_output_size, number_of_hidden_layers, nn.ReLU())
        self.q_criterion = nn.MSELoss()
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate_policy)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.learning_rate_q)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.learning_rate_q)

    def train_batch(self, observations, next_observations, actions, rewards, terminals):
        """
        Forward pass.
        Assumes inputs are torch tensors.
        """
        # actions from current policy
        policy_actions, policy_log_probability = self.policy(observations)
        policy_next_actions, policy_next_log_probability = self.policy(next_observations)

        # concatenate observations and corresponding actions
        observation_actions = torch.cat((observations, actions), dim=1)
        observation_policy_actions = torch.cat((observations, policy_actions), dim=1)
        next_observation_policy_next_actions = torch.cat((next_observations, policy_next_actions), dim=1)

        # q-values
        q1_policy_actions = self.q1(observation_policy_actions)
        q2_policy_actions = self.q2(observation_policy_actions)
        q_policy_actions = torch.min(q1_policy_actions, q2_policy_actions)
        q1_actions = self.q1(observation_actions)
        q2_actions = self.q2(observation_actions)

        # target q-values
        target_q1 = self.target_q1(next_observation_policy_next_actions)
        target_q2 = self.target_q2(next_observation_policy_next_actions)
        target_q = torch.min(target_q1, target_q2) - self.temperature * policy_next_log_probability
        q_target = self.reward_scale * rewards * (1.0 - terminals) * self.discount_factor * target_q
        q_target_detached = q_target.detach()

        # losses
        policy_loss = torch.mean(self.temperature * policy_log_probability - q_policy_actions)
        q1_loss = self.q_criterion(q1_actions, q_target_detached)
        q2_loss = self.q_criterion(q2_actions, q_target_detached)

        # optimize
        self.optimize(self.q1_optimizer, q1_loss)
        self.optimize(self.q2_optimizer, q2_loss)
        self.optimize(self.policy_optimizer, policy_loss)
        self.update_exponential_moving_target(self.q1, self.target_q1)
        self.update_exponential_moving_target(self.q2, self.target_q2)

        return policy_loss.detach().numpy(), q1_loss.detach().numpy(), q2_loss.detach().numpy()

    def get_action(self, observation):
        """
        Computes next action.
        Assumes input is a numpy array.
        """
        return self.policy.get_action(observation)

    def optimize(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update_exponential_moving_target(self, q, target):
        for q_param, target_param in zip(q.parameters(), target.parameters()):
            q_contribution = q_param.data * self.exponential_weight
            target_contribution = target_param.data * (1.0 - self.exponential_weight)
            new_target_param = target_contribution + q_contribution
            target_param.data.copy_(new_target_param)

    def load(self, path):
        raise NotImplementedError('model.load is not implemented yet')

    def save(self, path):
        raise NotImplementedError('model.save is not implemented yet')
