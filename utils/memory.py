import itertools
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity, n_step, discount=0.9):
        self.memory = deque([],maxlen=capacity)
        self.buffer = deque([],maxlen=n_step)
        self.n_step = n_step
        self.discount = discount

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(Transition(*args))

        if args[2] is None:
            while self.buffer:
                self.create_n_step_buffer()
                self.buffer.popleft()
        else:
            self.create_n_step_buffer()


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def create_n_step_buffer(self):
        total_discount = 1
        init_state, action, _, n_step_reward = self.buffer[0]
        for _, _, next_state, reward in itertools.islice(self.buffer, 1, None):
            total_discount *= self.discount
            n_step_reward += reward*total_discount
        _, _, final_state, _ = self.buffer[-1]

        self.memory.append(Transition(*(init_state, action, final_state, n_step_reward)))

    def __len__(self):
        return len(self.memory)