"""
This code is adopted from https://github.com/ikostrikov/pytorch-trpo
"""
from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample_n(self, n):
        n = min(n, len(self.memory))
        samples = random.sample(self.memory, n)
        states, actions, masks, next_states, rewards = zip(*samples)
        return Transition(states, actions, masks, next_states, rewards)

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
