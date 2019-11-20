from collections import namedtuple
from app.replay_buffer import ReplayBuffer
from app.types import Trajectory
import unittest

Options = namedtuple('Options', ['max_buffer_size', 'observation_size', 'action_size', 'batch_size'])

class TestReplayBuffer(unittest.TestCase):

    def test_does_not_exceed_max_size(self):
        options = Options(max_buffer_size=2, observation_size=1, action_size=1, batch_size=1)
        buffer = ReplayBuffer(options)
        self.assertEqual(buffer.buffer_size, 0)
        buffer.add_sample([1], [2], [1], 1, 0)
        self.assertEqual(buffer.buffer_size, 1)
        buffer.add_sample([2], [3], [0], 0, 0)
        self.assertEqual(buffer.buffer_size, 2)
        buffer.add_sample([3], [4], [0], 0, 1)
        self.assertEqual(buffer.buffer_size, 2)

    def test_stores_a_sample(self):
        options = Options(max_buffer_size=1, observation_size=1, action_size=1, batch_size=1)
        buffer = ReplayBuffer(options)
        buffer.add_sample([1], [2], [3], 4, 1)
        self.assertAlmostEqual(buffer.observations[0][0], 1)
        self.assertAlmostEqual(buffer.next_observations[0][0], 2)
        self.assertAlmostEqual(buffer.actions[0][0], 3)
        self.assertAlmostEqual(buffer.rewards[0], 4)
        self.assertEqual(buffer.terminals[0], 1)

    def test_top_is_advanced(self):
        options = Options(max_buffer_size=2, observation_size=1, action_size=1, batch_size=1)
        buffer = ReplayBuffer(options)
        self.assertEqual(buffer.top, 0)
        buffer.add_sample([1], [2], [1], 1, 0)
        self.assertEqual(buffer.top, 1)

    def test_top_resets_to_zero_when_reaching_max_size(self):
        options = Options(max_buffer_size=2, observation_size=1, action_size=1, batch_size=1)
        buffer = ReplayBuffer(options)
        buffer.add_sample([1], [2], [1], 1, 0)
        buffer.add_sample([2], [3], [0], 0, 0)
        self.assertEqual(buffer.top, 0)

    def test_add_trajectory(self):
        options = Options(max_buffer_size=1, observation_size=2, action_size=1, batch_size=1)
        buffer = ReplayBuffer(options)
        trajectory = Trajectory(observations=[[1, 1]], next_observations=[[2, 2]], actions=[[3]], rewards=[4], terminals=[0])
        buffer.add_trajectory(trajectory)
        self.assertAlmostEqual(buffer.observations[0][0], 1)
        self.assertAlmostEqual(buffer.observations[0][1], 1)
        self.assertAlmostEqual(buffer.next_observations[0][0], 2)
        self.assertAlmostEqual(buffer.next_observations[0][1], 2)
        self.assertAlmostEqual(buffer.actions[0], 3)
        self.assertAlmostEqual(buffer.rewards[0], 4)
        self.assertEqual(buffer.terminals[0], 0)

    def test_add_trajectories(self):
        options = Options(max_buffer_size=2, observation_size=1, action_size=1, batch_size=1)
        buffer = ReplayBuffer(options)
        trajectory1 = Trajectory(observations=[[1]], next_observations=[[2]], actions=[[3]], rewards=[4], terminals=[0])
        trajectory2 = Trajectory(observations=[[2]], next_observations=[[3]], actions=[[4]], rewards=[5], terminals=[1])
        buffer.add_trajectories([trajectory1, trajectory2])

        # trajectory 1
        self.assertAlmostEqual(buffer.observations[0][0], 1)
        self.assertAlmostEqual(buffer.next_observations[0][0], 2)
        self.assertAlmostEqual(buffer.actions[0][0], 3)
        self.assertAlmostEqual(buffer.rewards[0], 4)
        self.assertEqual(buffer.terminals[0], 0)

        # trajectory 2
        self.assertAlmostEqual(buffer.observations[1][0], 2)
        self.assertAlmostEqual(buffer.next_observations[1][0], 3)
        self.assertAlmostEqual(buffer.actions[1][0], 4)
        self.assertAlmostEqual(buffer.rewards[1], 5)
        self.assertEqual(buffer.terminals[1], 1)

    def test_random_batch(self):
        options = Options(max_buffer_size=1, observation_size=1, action_size=1, batch_size=1)
        buffer = ReplayBuffer(options)
        buffer.add_sample([1], [2], [3], 4, 1)
        batch = buffer.random_batch()
        self.assertAlmostEqual(batch.observation[0], 1)
        self.assertAlmostEqual(batch.next_observation[0], 2)
        self.assertAlmostEqual(batch.action[0], 3)
        self.assertAlmostEqual(batch.reward[0], 4)
        self.assertEqual(batch.terminal[0], 1)
