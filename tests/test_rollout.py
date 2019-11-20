from app.environments import TestEnvironment
from app.policies import TestPolicy
from app.rollout import rollout
from app.types import Trajectory
import numpy as np
import unittest

class TestRollout(unittest.TestCase):

    def test_rollout_reach_terminal(self):
        environment = TestEnvironment(0, 2)
        policy = TestPolicy(1)
        trajectory = rollout(environment, policy)
        expected_observations = np.array([[0], [1], [2]])
        expected_next_observations = np.array([[1], [2], [3]])
        expected_actions = np.array([[1], [1], [1]])
        expected_rewards = np.array([1, 1, -1])
        expected_terminals = np.array([0, 0, 1])
        np.testing.assert_array_equal(trajectory.observations, expected_observations)
        np.testing.assert_array_equal(trajectory.next_observations, expected_next_observations)
        np.testing.assert_array_equal(trajectory.actions, expected_actions)
        np.testing.assert_array_equal(trajectory.rewards, expected_rewards)
        np.testing.assert_array_equal(trajectory.terminals, expected_terminals)

    def test_rollout_terminate_at_max_length(self):
        environment = TestEnvironment(0, 2)
        policy = TestPolicy(1)
        trajectory = rollout(environment, policy, max_trajectory_length=2)
        expected_observations = np.array([[0], [1]])
        expected_next_observations = np.array([[1], [2]])
        expected_actions = np.array([[1], [1]])
        expected_rewards = np.array([1, 1])
        expected_terminals = np.array([0, 0])
        np.testing.assert_array_equal(trajectory.observations, expected_observations)
        np.testing.assert_array_equal(trajectory.next_observations, expected_next_observations)
        np.testing.assert_array_equal(trajectory.actions, expected_actions)
        np.testing.assert_array_equal(trajectory.rewards, expected_rewards)
        np.testing.assert_array_equal(trajectory.terminals, expected_terminals)
