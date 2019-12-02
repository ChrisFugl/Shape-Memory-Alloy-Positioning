from app.config import Config
from app.environments import get_environment
from app.model import Model
from app.policies import get_policy
from app.replay_buffer import ReplayBuffer
from app.rollout import rollouts
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch
import sys
import yaml
import yamale

CONFIG_SCHEMA_PATH = 'app/config/schema.yaml'


def main():
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    config = load_config(options.config)
    environment = get_environment(config.environment_type, config.environment)
    policy = get_policy(config.policy_type, config.policy, environment)
    model = Model(config.model, environment, policy)
    model.load(config.save_model)
    test(model, environment, config)

def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    options = parser.parse_args(arguments)
    return options


def load_config(config_path):
    schema = yamale.make_schema(CONFIG_SCHEMA_PATH)
    data = yamale.make_data(config_path)
    yamale.validate(schema, data, strict=True)
    config = Config(**data[0][0])
    return config


def test(model, environment, config):
    """
    Test the model according to the configuration and environment.

    :type model: app.model.Model
    :type environment: app.environments.environment.Environment
    :type config: app.config.Config
    """
    model.eval_mode()
    trajectories = rollouts(environment, model.policy, 1, max_trajectory_length=20)
    states = trajectories[0].observations
    rewards = trajectories[0].rewards
    print(f'start:          {states[0, 0]:0.02f}')
    print(f'final:          {states[-1, 0]:0.02f}')
    print(f'last 5:         {states[-5:, 0].mean():0.02f}')
    print(f'sum of rewards: {sum(rewards):0.02f}')


if __name__ == '__main__':
    main()
