from agents.base_agent import BaseAgent
from utils.memory import ReplayMemory
from utils.memory import Transition

import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math

class DQN(BaseAgent):

    def __init__(self, num_actions, network, gamma, batch_size, n_step, lr,
                 replay_mem_size, eps_start, eps_end, eps_decay, device='cuda', tensorboard=None):
        super().__init__(network)

        self.network = network
        self.target_network = copy.deepcopy(network)

        self.num_actions = num_actions
        self.replay_mem = ReplayMemory(capacity=replay_mem_size, n_step=n_step)
        self.optimizer = optim.Adam(network.parameters(), lr=lr)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = device
        self.writer = tensorboard
        self.steps_done = 1

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.network(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)

    def learn(self):
        if len(self.replay_mem) < self.batch_size:
            return
        transitions = self.replay_mem.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.writer is not None:
            self.writer.add_scalar("Loss/train", loss.mean(), self.steps_done)

    def remember(self, *args):
        self.replay_mem.push(*args)

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())
