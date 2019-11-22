from app.config.environments import TestEnvironmentConfig
from app.config.policies import TestPolicyConfig
from app.environments import TestEnvironment
from app.policies import TestPolicy
from app.rollout import rollout, rollouts
from app.types import Trajectory
import numpy as np
import unittest

test_environment_config = TestEnvironmentConfig(initial_state=0, final_state=2)
test_policy_config = TestPolicyConfig(change=1)

class TestRollout(unittest.TestCase):

    def test_rollout_reach_terminal(self):
        environment = TestEnvironment(test_environment_config)
        policy = TestPolicy(test_policy_config, environment)
        trajectory = rollout(environment, policy)
        expected_observations = np.array([[0], [1]])
        expected_next_observations = np.array([[1], [2]])
        expected_actions = np.array([[1], [1]])
        expected_rewards = np.array([1, 1])
        expected_terminals = np.array([0, 1])
        np.testing.assert_array_equal(trajectory.observations, expected_observations)
        np.testing.assert_array_equal(trajectory.next_observations, expected_next_observations)
        np.testing.assert_array_equal(trajectory.actions, expected_actions)
        np.testing.assert_array_equal(trajectory.rewards, expected_rewards)
        np.testing.assert_array_equal(trajectory.terminals, expected_terminals)

    def test_rollout_terminate_at_max_length(self):
        environment = TestEnvironment(test_environment_config)
        policy = TestPolicy(test_policy_config, environment)
        trajectory = rollout(environment, policy, max_trajectory_length=1)
        expected_observations = np.array([[0]])
        expected_next_observations = np.array([[1]])
        expected_actions = np.array([[1]])
        expected_rewards = np.array([1])
        expected_terminals = np.array([0])
        np.testing.assert_array_equal(trajectory.observations, expected_observations)
        np.testing.assert_array_equal(trajectory.next_observations, expected_next_observations)
        np.testing.assert_array_equal(trajectory.actions, expected_actions)
        np.testing.assert_array_equal(trajectory.rewards, expected_rewards)
        np.testing.assert_array_equal(trajectory.terminals, expected_terminals)

    def test_rollouts(self):
        environment = TestEnvironment(test_environment_config)
        policy = TestPolicy(test_policy_config, environment)
        trajectories = rollouts(environment, policy, 2, max_trajectory_length=1)
        self.assertEqual(len(trajectories), 2)
        expected_trajectory1 = Trajectory(
            observations=np.array([[0]]),
            next_observations=np.array([[1]]),
            actions=np.array([[1]]),
            rewards=np.array([1]),
            terminals=np.array([0])
        )
        expected_trajectory2 = Trajectory(
            observations=np.array([[0]]),
            next_observations=np.array([[1]]),
            actions=np.array([[1]]),
            rewards=np.array([1]),
            terminals=np.array([0])
        )
        np.testing.assert_array_equal(trajectories[0].observations, expected_trajectory1.observations)
        np.testing.assert_array_equal(trajectories[0].next_observations, expected_trajectory1.next_observations)
        np.testing.assert_array_equal(trajectories[0].actions, expected_trajectory1.actions)
        np.testing.assert_array_equal(trajectories[0].rewards, expected_trajectory1.rewards)
        np.testing.assert_array_equal(trajectories[0].terminals, expected_trajectory1.terminals)
        np.testing.assert_array_equal(trajectories[1].observations, expected_trajectory2.observations)
        np.testing.assert_array_equal(trajectories[1].next_observations, expected_trajectory2.next_observations)
        np.testing.assert_array_equal(trajectories[1].actions, expected_trajectory2.actions)
        np.testing.assert_array_equal(trajectories[1].rewards, expected_trajectory2.rewards)
        np.testing.assert_array_equal(trajectories[1].terminals, expected_trajectory2.terminals)
