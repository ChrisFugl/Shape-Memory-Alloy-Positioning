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

CONFIG_SCHEMA_PATH = 'app/config/schema.yaml'


def main():
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    config = load_config(options.config)
    exploration_environment = get_environment(config.environment_type, config.environment)
    evaluation_environment = get_environment(config.environment_type, config.environment)
    policy = get_policy(config.policy_type, config.policy, evaluation_environment)
    variant = dict(
        algorithm='SAC',
        version='normal',
        layer_size=config.model.network.hidden_size,
        replay_buffer_size=config.max_buffer_size,
        algorithm_kwargs=dict(
            collect_actions=config.collect_actions,
            collect_actions_every=config.collect_actions_every,
            num_epochs=config.iterations,
            num_eval_steps_per_epoch=config.environment_steps,
            num_trains_per_train_loop=config.gradient_steps,
            num_expl_steps_per_train_loop=config.environment_steps,
            min_num_steps_before_training=1,
            max_path_length=config.max_trajectory_length,
            batch_size=config.batch_size,
        ),
        trainer_kwargs=dict(
            discount=config.model.discount_factor,
            soft_target_tau=config.model.exponential_weight,
            target_update_period=1,
            policy_lr=config.model.learning_rate_policy,
            qf_lr=config.model.learning_rate_q,
            reward_scale=config.model.reward_scale,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger(options.name, variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, exploration_environment, evaluation_environment, policy)


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument(
        '-name',
        type=str,
        default='baseline',
        help='name of experiment, it is used for tensorboard logging')
    options = parser.parse_args(arguments)
    return options


def load_config(config_path):
    schema = yamale.make_schema(CONFIG_SCHEMA_PATH)
    data = yamale.make_data(config_path)
    yamale.validate(schema, data, strict=True)
    config = Config(**data[0][0])
    return config


def experiment(variant, expl_env, eval_env, policy):
    expl_env = NormalizedBoxEnv(expl_env)
    eval_env = NormalizedBoxEnv(eval_env)
    obs_dim = expl_env.observation_size
    action_dim = eval_env.action_size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
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
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == '__main__':
    main()
