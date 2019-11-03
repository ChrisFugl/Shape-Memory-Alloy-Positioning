import configargparse
import yaml

def main():
    options = parse_arguments()
    # TODO: start training

def parse_arguments():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--config', required=False, is_config_file=True, help='config file path')
    parser.add('--save_config', required=False, default=None, type=str, help='path of config file where arguments can be saved')
    parser.add('--save_model', required=False, default=None, type=str, help='path of file to save trained model')
    parser.add('--environment', required=True, choices=['real_time', 'simulated'], help='which environment to use (real_time or simulated)')
    parser.add('--iterations', required=False, default=100000, type=int, help='number of iterations')
    parser.add('--environment_steps', required=False, default=100, type=int, help='number of environment steps to train for')
    parser.add('--gradient_steps', required=False, default=10, type=int, help='number of gradient steps to train for')
    parser.add('--temperature', required=False, default=0.1, type=float, help='weight multiplied to entropy in objective (alpha in SAC paper, default 0.1)')
    parser.add('--learning_rate_value', required=False, default=0.001, type=float, help='learning rate for value function approximator (default 0.001)')
    parser.add('--learning_rate_q', required=False, default=0.001, type=float, help='learning rate for Q-function approximator (default 0.001)')
    parser.add('--learning_rate_policy', required=False, default=0.001, type=float, help='learning rate for policy function (default 0.001)')
    parser.add('--exponential_weight', required=False, default=0.9, type=float, help='weight in exponential moving average of value weights (0 < tau <= 1, default 0.9)')
    parser.add('--discount_factor', required=False, default=0.99, type=float, help='discount factor (0 < gamma <= 1, default 0.99)')
    options = parser.parse_args()

    # remove keys that should not be saved to config file
    save_config = options.save_config
    del options.config
    del options.save_config

    # save config file
    if save_config is not None:
        with open(save_config, 'w') as config_file:
            yaml.dump(vars(options), config_file)

    return options

if __name__ == '__main__':
    main()
