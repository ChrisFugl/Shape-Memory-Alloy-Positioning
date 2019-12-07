import abc

import gtimer as gt
import itertools
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector

import time
import os
import pickle
import torch

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            collect_actions,
            collect_actions_every,
            save_dir,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            save_checkpoint_interval_s=1800,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            collect_actions,
            collect_actions_every,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.save_time = save_checkpoint_interval_s
        self.save_dir = save_dir

    def _train(self):
        start_time = time.time()

        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        # interpret a negative number of epochs as an infinite training (until manually stopped)
        if self.num_epochs < 0:
            iterable_epoch = itertools.count(self._start_epoch)
        else:
            iterable_epoch = range(self._start_epoch, self.num_epochs)

        for epoch in gt.timed_for(iterable_epoch, save_itrs=True):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            cur_time = time.time()
            if cur_time - start_time > self.save_time:
                ckp_path = os.path.join(self.save_dir, 'iteration_' + str(epoch))
                if not os.path.exists(ckp_path):
                    os.mkdir(ckp_path)
                model_checkpoint = self.trainer.get_checkpoint()
                torch.save(model_checkpoint, os.path.join(ckp_path, 'model.pt'))
                replay_dict = self.replay_buffer.to_dict()
                replay_dict['iteration'] = epoch
                with open(os.path.join(ckp_path, 'replay_buffer.pkl'), 'wb') as f:
                    pickle.dump(replay_dict, f)
                start_time = time.time()

            self._end_epoch(epoch)
