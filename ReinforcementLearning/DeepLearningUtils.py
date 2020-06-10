import numpy as np
from numpy import random as rnd
import random
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
from torch.nn.utils.rnn import pad_sequence
import progressbar

# just a feed forward neural network to estimate Q(s,a) values
class DQN(nn.Module):
    def __init__(self, envstate_dim, action_dim):
        super(DQN, self).__init__()
        self.input_dim = envstate_dim
        self.output_dim = action_dim

        self.ff = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 124),
            nn.ReLU(),
            nn.Linear(124, self.output_dim),)

    def forward(self, state):
        state = torch.FloatTensor(state).float()
        qvals = self.ff(state)
        return qvals


# replay buffers implemented as lists. this is actually recommended by Python over Deques for random retrieval
class Buffer():
    def __init__(self):
        self.buffer = []

    def size(self):
        return len(self.buffer)

    # add a memory
    def push(self,state,action,new_state,reward):
        experience = (state,action,new_state,reward)
        self.buffer.append(experience)

    # take a random sample to perform learning on decorrelated transitions
    def sample(self,batch_size):
        batchSample = random.sample(self.buffer,batch_size)
        # now need to put everyone in the correct columns
        state_batch = []
        action_batch = []
        new_state_batch = []
        reward_batch = []

        # prepare the batch sample for training
        for experience in batchSample:
            state,action,new_state,reward = experience
            state_batch.append(state)
            action_batch.append(action)
            new_state_batch.append(new_state)
            reward_batch.append(reward)
        return (state_batch, action_batch, reward_batch, new_state_batch)
