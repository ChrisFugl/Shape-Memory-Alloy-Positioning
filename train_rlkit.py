from app.config import Config
from app.environments import get_environment
from app.policies import get_policy
import argparse
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import sys
import yamale
import os
import torch
import pickle

CONFIG_SCHEMA_PATH = 'app/config/schema.yaml'
CHECKPOINTS_DIR = 'checkpoints/'

def main():
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    config = load_config(options.config)
    load_iter = options.load
    exploration_environment = get_environment(config.environment_type, config.environment)
    evaluation_environment = get_environment(config.environment_type, config.environment)
    policy = get_policy(config.policy_type, config.policy, evaluation_environment)
    variant = dict(
        algorithm='SAC',
        version='normal',
        layer_size=config.model.network.hidden_size,
        number_of_layers=config.model.network.number_of_hidden_layers,
        replay_buffer_size=config.max_buffer_size,
        algorithm_kwargs=dict(
            batch_size=config.batch_size,
            collect_actions=config.collect_actions,
            collect_actions_every=config.collect_actions_every,
            num_epochs=config.iterations,
            num_eval_steps_per_epoch=config.evaluation_steps,
            num_trains_per_train_loop=config.gradient_steps,
            num_expl_steps_per_train_loop=config.exploration_steps,
            min_num_steps_before_training=config.min_num_steps_before_training,
            max_path_length=config.max_trajectory_length,
            save_checkpoint_interval_s=config.save_checkpoint_interval_s,
        ),
        trainer_kwargs=dict(
            discount=config.model.discount_factor,
            soft_target_tau=config.model.exponential_weight,
            target_update_period=config.model.target_update_period,
            policy_lr=config.model.learning_rate_policy,
            qf_lr=config.model.learning_rate_q,
            reward_scale=config.model.reward_scale,
            use_automatic_entropy_tuning=config.model.use_automatic_entropy_tuning,
        ),
    )
    setup_logger(options.name, variant=variant, tensorboard_log_dir=f'runs/{options.name}')
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)

    ckp_path = os.path.join(CHECKPOINTS_DIR, options.name)
    os.makedirs(ckp_path, exist_ok=True)

    experiment(variant, exploration_environment, evaluation_environment, policy, ckp_path, load_iter)


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--load', default=None, required=False, type=str, help='data from given iteration will be loaded')
    parser.add_argument('--name', type=str, default='baseline', help='name of experiment, it is used for tensorboard logging')
    options = parser.parse_args(arguments)
    return options


def load_config(config_path):
    schema = yamale.make_schema(CONFIG_SCHEMA_PATH)
    data = yamale.make_data(config_path)
    yamale.validate(schema, data, strict=True)
    config = Config(**data[0][0])
    return config


def experiment(variant, expl_env, eval_env, policy, checkpoint_dir, load_iter):
    print(f'Environment: {expl_env.name}')
    print(f'Policy: {policy.name}')

    expl_env = NormalizedBoxEnv(expl_env)
    eval_env = NormalizedBoxEnv(eval_env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if load_iter is not None:
        load_dir = os.path.join(checkpoint_dir, 'iteration_' + load_iter)

        qf1 = torch.load(os.path.join(load_dir, 'q1'))
        qf2 = torch.load(os.path.join(load_dir, 'q2'))
        target_qf1 = torch.load(os.path.join(load_dir, 'target_q1'))
        target_qf2 = torch.load(os.path.join(load_dir, 'target_q2'))
        policy = torch.load(os.path.join(load_dir, 'policy'))
        eval_policy = MakeDeterministic(policy)
        eval_path_collector = MdpPathCollector(eval_env, eval_policy)
        expl_path_collector = MdpPathCollector(expl_env, policy)

        with open(os.path.join(load_dir, 'replay_buffer.p'), 'rb') as f:
            attr_dict = pickle.load(f)
        replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'], expl_env, attr_dict)

        trainer = SACTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['trainer_kwargs']
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            save_dir=checkpoint_dir,
            **variant['algorithm_kwargs']
        )
        start_epoch = attr_dict['iteration'] + 1
    else:
        layer_size = variant['layer_size']
        number_of_layers = variant['number_of_layers']
        hidden_sizes = [layer_size] * number_of_layers
        q_input_size = obs_dim + action_dim
        q_output_size = 1
        qf1 = FlattenMlp(input_size=q_input_size, output_size=q_output_size, hidden_sizes=hidden_sizes)
        qf2 = FlattenMlp(input_size=q_input_size, output_size=q_output_size, hidden_sizes=hidden_sizes)
        target_qf1 = FlattenMlp(input_size=q_input_size, output_size=q_output_size, hidden_sizes=hidden_sizes)
        target_qf2 = FlattenMlp(input_size=q_input_size, output_size=q_output_size, hidden_sizes=hidden_sizes)
        eval_policy = MakeDeterministic(policy)
        eval_path_collector = MdpPathCollector(eval_env, eval_policy)
        expl_path_collector = MdpPathCollector(expl_env, policy)
        replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'], expl_env)
        trainer = SACTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **variant['trainer_kwargs']
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            save_dir=checkpoint_dir,
            **variant['algorithm_kwargs']
        )
        start_epoch = 0
    algorithm.to(ptu.device)
    algorithm.train(start_epoch=start_epoch)

if __name__ == '__main__':
    main()
