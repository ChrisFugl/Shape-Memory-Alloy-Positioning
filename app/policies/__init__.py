from app.policies.gaussian import GaussianPolicy
from app.policies.test import TestPolicy

_policy_classes = {
    'gaussian': GaussianPolicy,
    'test': TestPolicy
}

def get_policy(policy_type, policy_config, environment, **kwargs):
    """
    :type policy_type: str
    :type policy_config: app.config.policies.PolicyConfig
    :type environment: app.config.environments.EnvironmentConfig
    """
    policy_class = _policy_classes[policy_type]
    return policy_class(policy_config, environment, **kwargs)
