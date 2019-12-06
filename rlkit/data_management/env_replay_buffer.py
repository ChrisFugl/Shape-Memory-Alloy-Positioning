from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np

import _pickle as pickle


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            attr_dict = None,
            env_info_sizes=None,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """

        # self._attr_dict = attr_dict

        # if self._attr_dict:
        #     self.env = self._attr_dict['env']
        #     self._ob_space = self._attr_dict['ob_space']
        #     self._action_space = self.attr_dict['action_space']

        if attr_dict:
            self.env = attr_dict['env']
            self._ob_space = attr_dict['ob_space']
            self._action_space = attr_dict['action_space']

        else:
            self.env = env
            self._ob_space = env.observation_space
            self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            attr_dict = attr_dict
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

    def to_dict(self):
        
        attr_dict = {}

        attr_dict['env'] = self.env
        attr_dict['ob_space'] = self._ob_space
        attr_dict['action_space'] = self._action_space
        attr_dict['observation_dim'] = self._observation_dim
        attr_dict['action_dim'] = self._action_dim
        attr_dict['max_replay_buffer_size'] = self._max_replay_buffer_size
        attr_dict['observations'] = self._observations
        attr_dict['next_obs'] = self._next_obs
        attr_dict['actions'] = self._actions
        attr_dict['rewards'] = self._rewards
        attr_dict['terminals'] = self._terminals
        attr_dict['top'] = self._top
        attr_dict['size'] = self._size

        return attr_dict














