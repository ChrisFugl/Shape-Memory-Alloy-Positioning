import arguments
import configargparse
import copy
import yaml

def main():
    options = parse_arguments()
    # TODO: start training

def parse_arguments():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('environment', choices=['real_time', 'simulated'], help='which environment to use')

    parser.add('--config', required=False, is_config_file=True, help='config file path')
    parser.add('--save_config', required=False, default=None, type=str, help='path of config file where arguments can be saved')
    parser.add('--save_model', required=False, default=None, type=str, help='path of file to save trained model')
    arguments.add_soft_actor_critic_arguments(parser)
    arguments.add_simulated_environment_arguments(parser)
    options = parser.parse_args()

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

if __name__ == '__main__':
    main()
