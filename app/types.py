from collections import namedtuple

Batch = namedtuple('Batch', [
    'observation',
    'next_observation',
    'action',
    'reward',
    'terminal'
])

Trajectory = namedtuple('Trajectory', [
    'observations',
    'next_observations',
    'actions',
    'rewards',
    'terminals'
])
