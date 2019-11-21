from collections import namedtuple

Batch = namedtuple('Batch', [
    'observations',
    'next_observations',
    'actions',
    'rewards',
    'terminals'
])

Trajectory = namedtuple('Trajectory', [
    'observations',
    'next_observations',
    'actions',
    'rewards',
    'terminals'
])
