from app.types import Trajectory
import numpy as np

def rollout(environment, policy, max_trajectory_length=np.inf):
    """
    Makes rollsouts according to the policy and the environment.

    :param environment: subclass of app.environment.Environment
    :param policy: current policy of the agent
    :param max_trajectory_length: optional max length of the trajectory (default infinite)
    :return: Trajectory
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    observation = environment.get_state()
    next_observation = None
    trajectory_length = 0
    while trajectory_length < max_trajectory_length:
        action = policy.get_action(observation)
        next_observation, reward, terminal = environment.step(action)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        if isinstance(terminal, bool):
            terminal = int(terminal)
        terminals.append(terminal)
        trajectory_length += 1
        if terminal:
            break
        observation = next_observation
    actions = actions2numpy(actions)
    observations, next_observations = observations2numpy(observations, next_observation)
    rewards = np.array(rewards, dtype=np.float64)
    terminals = np.array(terminals, dtype=np.uint8)
    return Trajectory(observations, next_observations, actions, rewards, terminals)

def actions2numpy(actions):
    np_actions = np.array(actions, dtype=np.float64)
    if len(np_actions) == 1:
        np_actions = np.expand_dims(np_actions, 1)
    return np_actions

def observations2numpy(observations, last_observation):
    np_observations = np.array(observations, dtype=np.float64)
    if len(np_observations) == 1:
        np_observations = np.expand_dims(np_observations, 1)
        last_observation = [last_observation]
    np_last_observation = np.expand_dims(last_observation, 0)
    np_next_observations = np.vstack((np_observations[1:, :], np_last_observation))
    return np_observations, np_next_observations
