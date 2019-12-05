from app.policies.categorical import CategoricalPolicy
from app.policies.gaussian import GaussianPolicy
from app.policies.range import RangePolicy
from app.policies.test import TestPolicy
from app.policies.tanhgaussian import TanhGaussianPolicy

_policy_classes = {
    'categorical': CategoricalPolicy,
    'gaussian': GaussianPolicy,
    'range': RangePolicy,
    'test': TestPolicy,
    'tanhgaussian': TanhGaussianPolicy
}

def get_policy(policy_type, policy_config, environment, **kwargs):
    """
    :type policy_type: str
    :type policy_config: app.config.policies.PolicyConfig
    :type environment: app.config.environments.EnvironmentConfig
    """
    policy_class = _policy_classes[policy_type]
    return policy_class(policy_config, environment, **kwargs)
