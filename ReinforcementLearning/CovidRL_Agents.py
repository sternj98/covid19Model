import numpy as np
from numpy import random as rnd
import random
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import progressbar
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
import progressbar

from PopulationClasses import Population
from DeepLearningUtils import DQN, Buffer

# a class for agents that use feedforward neural networks to calculate Q(s,a)
class DeepQAgent():
    def __init__(self,state_dim,action_dim):
        self.policy_net = DQN(state_dim,action_dim) # network used to calculate policy
        self.target_net = DQN(state_dim,action_dim) # network used to calculate target
        self.target_net.eval() # throw that baby in eval mode because we don't care about its gradients
        self.target_update = 50 # update our target network every 50 timesteps
        self.replay_buffer = Buffer() # replay buffer implemented as a list
        self.action_dim = action_dim

        self.eps_start = 0.1 # initial exploration rate
        self.eps_end = 0.95 # ultimate exploration value
        self.eps_decay = 300 # decay parameter for exploration rate
        self.epsilon = self.eps_start # initialize epsilon

        self.gamma = 0.99 # discount

#         self.optimizer = torch.optim.SGD(self.policy_net.parameters(),lr=0.001, momentum=0.9)
#         self.optimizer = torch.optim.RMSprop(self.policy_net.parameters()) # experiment w/ different optimizers
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.huber_loss = F.smooth_l1_loss

    def select_action(self,state):
        state = torch.FloatTensor(state).float()
        if rnd.rand() < self.epsilon: # greedy action
            with torch.no_grad():
                qvals = self.policy_net.forward(state) # forward run through the policy network
                action = np.argmax(qvals.detach().numpy()) # need to detach from auto_grad before sending to numpy
        else:
            action = random.choice(list(range(self.action_dim)))
        return action

    def update(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)

        self.optimizer.zero_grad() # zero_grad before computing loss

        loss = self.compute_loss(batch)

        loss.backward() # get the gradients

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # gradient clipping

        self.optimizer.step() # backpropagate

        return loss

    def compute_loss(self, batch):
        states, actions, rewards, next_states = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

#         print(states.shape)
        curr_Q = self.policy_net.forward(states).gather(1,actions.unsqueeze(1)) # calculate the current Q(s,a) estimates
        next_Q = self.target_net.forward(next_states) # calculate Q'(s,a) (EV)
        max_next_Q = torch.max(next_Q,1)[0] # equivalent of taking a greedy action
        expected_Q = rewards + self.gamma * max_next_Q # Calculate total Q(s,a)

        loss = self.huber_loss(curr_Q, expected_Q.unsqueeze(1)) # unsqueeze is really important here to match dims!
        return loss
