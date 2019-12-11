from app.config import Config
from app.environments import get_environment
from app.policies import get_policy
import argparse
from copy import deepcopy
import numpy as np
import os
import pandas as pd
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import MakeDeterministic
import sys
import time
import torch
import yamale

CONFIG_SCHEMA_PATH = 'app/config/schema.yaml'

def main():
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    config = load_config(options.config)
    checkpoint_directory = os.path.join(config.checkpoint_dir, options.name)
    checkpoint_iteration_directory = os.path.join(checkpoint_directory, f'iteration_{options.load}')
    checkpoint = torch.load(os.path.join(checkpoint_iteration_directory, 'model.pt'))
    environment = make_environment(config, options)
    policy = get_policy(config.policy_type, config.policy, environment)
    policy.load_state_dict(checkpoint['policy'])
    policy = MakeDeterministic(policy)
    result_directory = os.path.join('results', options.name)
    os.makedirs(result_directory, exist_ok=True)
    evaluate(environment, policy, result_directory, config, options)
    save_options(options)


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--load', required=True, type=str, help='model from checkpoint at given iteration will be loaded')
    parser.add_argument('--name', required=True, type=str, default='baseline', help='name of experiment, it is used for saving test results')
    parser.add_argument('--num_of_trajectories', required=False, type=int, default=5, help='number of trajectories to evaluate against (default: 5)')
    parser.add_argument('--trajectory_length', required=False, type=int, default=128, help='number of steps in each trajectory (default: 128)')
    parser.add_argument('--goals', required=True, type=float, nargs='+', help='target goal positions in meters')
    options = parser.parse_args(arguments)
    return options


def load_config(config_path):
    schema = yamale.make_schema(CONFIG_SCHEMA_PATH)
    data = yamale.make_data(config_path)
    yamale.validate(schema, data, strict=True)
    config = Config(**data[0][0])
    return config


def evaluate(environment, policy, result_directory, environment):
    for goal in options.goals:
        goal_directory = os.path.join(result_directory, f'goal_{goal}')
        os.makedirs(goal_directory, exist_ok=True)
        environment._goal_start_position = goal
        for trajectory_index in range(options.num_of_trajectories):
            trajectory_dataframe = rollout(environment, policy, options)
            trajectory_path = os.path.join(goal_directory, f'trajectory_{trajectory_index}.csv')
            trajectory_dataframe.to_csv(trajectory_path, index=False)


def make_environment(config, options):
    environment_config = deepcopy(config.environment)
    environment_config.goal_type = 'static'
    environment = get_environment(config.environment_type, environment_config)
    environment = NormalizedBoxEnv(environment)
    return environment


TRAJECTORY_DTYPES = [
    np.int,
    np.float,
    np.float,
    np.float,
    np.float,
    np.float,
    np.float,
    np.float,
    np.float,
    np.float,
    np.float,
    np.float,
    np.bool,
]


TRAJECTORY_COLUMNS = [
    'timestep',
    'time_since_reset',
    'time_since_last_step',
    'position',
    'temperature',
    'voltage_min',
    'voltage_max',
    'action',
    'action_scaled',
    'velocity',
    'similarity',
    'reward',
    'terminal',
]


def rollout(environment, agent, options):
    dataframe = pd.DataFrame([], columns=TRAJECTORY_COLUMNS, dtype=TRAJECTORY_DTYPES)

    o = env.reset()
    # wait two seconds in between each new trajectory as to make sure
    # that the SMA will start in stable states
    time.sleep(2)
    # reset again to reset the timestamps in the environment
    o = env.reset()
    agent.reset()
    reset_timestamp = time.time()
    time_since_reset = 0.0
    next_o = None
    for timestep in range(1, options.trajectory_length + 1):
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        dataframe.loc[timestep - 1] = [
            timestep,
            time_since_reset,
            o[4], # time since last step
            o[2], # position
            o[0], # temperature
            o[6], # voltage min
            o[7], # voltage max
            env_info['action_before_scaling'],
            a,
            env_info['velocity'],
            env_info['similarity'],
            r,
            d,
        ]
        if d:
            break
        time_since_reset = time.time() - reset_timestamp
        o = next_o
    return dataframe


def save_options(options):
    pass


if __name__ == '__main__':
    main()
