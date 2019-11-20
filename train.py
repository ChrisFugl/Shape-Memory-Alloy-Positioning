from app.arguments import add_goal_arguments, add_soft_actor_critic_arguments, add_simulated_environment_arguments
from app.environments import RealTimeEnvironment, SimulatedEnvironment
import configargparse
import copy
import sys
import yaml

def main():
    arguments = sys.argv[1:]
    options = parse_arguments(arguments)
    environment = get_environment(options)
    model = train(options, environment)

    # save trained model
    if options.save_model is not None:
        model.save(options.save_model)

def parse_arguments(arguments):
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('environment', choices=['real_time', 'simulated'], help='which environment to use')

    parser.add('--config', required=False, is_config_file=True, help='config file path')
    parser.add('--save_config', required=False, default=None, type=str, help='path of config file where arguments can be saved')
    parser.add('--save_model', required=False, default=None, type=str, help='path of file to save trained model')
    add_goal_arguments(parser)
    add_soft_actor_critic_arguments(parser)
    add_simulated_environment_arguments(parser)
    options = parser.parse_args(arguments)

    # remove keys that should not be saved to config file
    save_options = copy.deepcopy(options)
    save_config = options.save_config
    del save_options.environment
    del save_options.config
    del save_options.save_config

    # save config file
    if save_config is not None:
        with open(save_config, 'w') as config_file:
            yaml.dump(vars(save_options), config_file)

    return options

def get_environment(options):
    if options.environment == 'real_time':
        return RealTimeEnvironment(options)
    else:
        return SimulatedEnvironment(options)

def train(options, environment):
    """
    Creates a model and trains the model according to the options and environment.

    :param options: hyperparameters
    :param environment: the environment to act in
    :return: trained model
    """
    # TODO: initialize model:
    #   * initialize value function parameters
    #   * initialize Q-function 1 parameters
    #   * initialize Q-function 2 parameters
    #   * initialize policy parameters
    #   * initialize exponential moving average of value function parameters
    # TODO: initialize replay buffer

    model = None
    # model = Model(options)
    # replay_buffer = ReplayBuffer(options)
    for iteration in range(options.iterations):
        for environment_step in range(options.environment_steps):
            # TODO: act in envionment
            #   * state = environment.get_state()
            #   * action = get action from model
            #   * next_state, reward = environment.step(action)
            #   * add (state, action, reward, next_state) to replay buffer
            pass
        for gradient_step in range(options.gradient_steps):
            # TODO: optimize neural networks based on gradients:
            #   * update value function parameters
            #   * update Q-function 1 parameters
            #   * update Q-function 2 parameters
            #   * update policy parameters
            #   * update exponential moving average of value function parameters
            pass
    return model

if __name__ == '__main__':
    main()
