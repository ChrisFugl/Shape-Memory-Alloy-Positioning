from rlkit.torch.sac.policies import TanhGaussianPolicy as RLKitTanhGaussianPolicy

class TanhGaussianPolicy(RLKitTanhGaussianPolicy):
    """
    Policy that should only be used for testing purposes.
    """

    name = 'tanh_gaussian'

    def __init__(self, config, environment):
        """
        :type config: app.config.policies.TanhGaussianPolicyConfig
        :type environment: app.config.environments.EnvironmentConfig
        """
        obs_dim = environment.observation_size
        action_dim = environment.action_size
        network = config.network
        hidden_sizes = [network.hidden_size] * network.number_of_hidden_layers
        super(TanhGaussianPolicy, self).__init__(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden_sizes)