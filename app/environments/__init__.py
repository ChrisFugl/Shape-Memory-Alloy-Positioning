from app.environments.real_time import RealTimeEnvironment
from app.environments.simulated import SimulatedEnvironment
from app.environments.test import TestEnvironment

_environment_classes = {
    'real_time': RealTimeEnvironment,
    'simulated': SimulatedEnvironment,
    'test': TestEnvironment
}

def get_environment(environment_type, environment_config):
    """
    :type environment_type: str
    :type environment_config: app.config.environments.EnvironmentConfig
    """
    environment_class = _environment_classes[environment_type]
    return environment_class(environment_config)
