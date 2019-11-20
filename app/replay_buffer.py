from app.types import Batch
import numpy as np

class ReplayBuffer:

    def __init__(self, options):
        """
        Creates a replay buffer for storing and sampling observed samples.

        :param options: a named tuple which should contain at least the following keys:
            * max_buffer_size
            * observation_size
            * action_size
            * batch_size
        """
        self.max_buffer_size = options.max_buffer_size
        self.observation_size = options.observation_size
        self.action_size = options.action_size
        self.batch_size = options.batch_size

        self.observations = np.zeros((options.max_buffer_size, options.observation_size), dtype=np.float64)
        self.next_observations = np.zeros((options.max_buffer_size, options.observation_size), dtype=np.float64)
        self.actions = np.zeros((options.max_buffer_size, options.action_size), dtype=np.float64)
        self.rewards = np.zeros((options.max_buffer_size, 1), dtype=np.float64)
        self.terminals = np.zeros((options.max_buffer_size, 1), dtype=np.uint8)

        self.top = 0
        self.buffer_size = 0

    def add_sample(self, observation, next_observation, action, reward, terminal):
        """
        Adds a sample to the store.

        :param observation: observation_size np.ndarray
        :param next_observation: observation_size np.ndarray
        :param action: action_size np.ndarray
        :param reward: np.float64
        :param terminal: unsigned integer (np.uint8])
        """
        self.observations[self.top] = observation
        self.next_observations[self.top] = next_observation
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminals[self.top] = terminal

        # set top to next position in buffer
        self.top = (self.top + 1) % self.max_buffer_size

        # increment size of buffer
        if self.buffer_size < self.max_buffer_size:
            self.buffer_size += 1

    def add_trajectory(self, trajectory):
        """
        Adds a trajectory to the store.

        :param trajectory: Trajectory from app.rollout.Rollout
        """
        for (observation, next_observation, action, reward, terminal) in zip(
            trajectory.observations,
            trajectory.next_observations,
            trajectory.actions,
            trajectory.rewards,
            trajectory.terminals
        ):
            self.add_sample(observation, next_observation, action, reward, terminal)

    def add_trajectories(self, trajectories):
        """
        Adds several trajectories to the store.

        :param trajectories: list of Trajectories
        """
        for trajectory in trajectories:
            self.add_trajectory(trajectory)

    def random_batch(self):
        indices = np.random.randint(0, self.buffer_size, self.batch_size)
        return Batch(
            observation=self.observations[indices],
            next_observation=self.next_observations[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            terminal=self.terminals[indices],
        )
