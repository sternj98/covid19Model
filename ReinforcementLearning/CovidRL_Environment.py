import numpy as np
from numpy import random as rnd
import random
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import progressbar

from PopulationClasses import Population

# defining constant integers to make labeling easy
SUS = 0 # susceptible
INF = 1 # infected
REC = 2 # recovered
DEAD = 3 # dead

# defining an age distribution
age_dist= np.array([6.0, 6.1, 6.5, 6.6, 6.6, 7.1, 6.7, 6.6, 6.1, 6.3, 6.4, 6.6, 6.3, 5.2, 4.1, 2.9, 1.9, 1.9])
under20 = np.sum(age_dist[0:4])
twentyto45 = np.sum(age_dist[4:9])
forty5to55 = np.sum(age_dist[9:11])
fifty5to65 = np.sum(age_dist[11:13])
sixty5to75 = np.sum(age_dist[13:15])
seventy5to85 = np.sum(age_dist[15:17])
eighty5plus = np.sum(age_dist[17:len(age_dist)])

#arranged ages according to the intervals set for death rates
age_dist = np.array([under20,twentyto45,forty5to55,fifty5to65,sixty5to75,seventy5to85,eighty5plus])
age_dist = age_dist / np.sum(age_dist) # probability distribution over ages


class DiseaseEnvironment():
    """
        An environment for control of exponential systems
    """
    def __init__(self,trial_len,pop_size,I0,p_connect,age_dist,p_infect):
        self.trial_len = trial_len
        self.pop_size = pop_size
        self.I0 = I0
        self.p_connect = p_connect
        self.age_dist = age_dist
        # can also have these on a distribution
        self.p_infect = p_infect

        self.terminal_state = -1
        self.episode_complete = False

        self.interact_rew = .1
        self.infect_penalty = .1
        self.dead_penalty = 5.
        self.test_penalty = .1

        self.reset()

    def reset(self): # new run of infection
        self.episode_complete = False
        self.day = 0
        # initialize a new population
        self.population = Population(self.pop_size,self.age_dist,self.I0,self.p_connect)
        population.prepSimulation(self.trial_len,self.p_infect)

    def execute_action(self, action): # execute the agent's action and return reward
        n_tests, policy = action # unpack action
        self.day += 1
        self.population.step(self.day,n_tests,policy)

        rew = self.interact_rew * policy * self.pop_size
              - self.infect_penalty * self.population.nInf[day]
              - self.dead_penalty * self.population.nDead[day]
              - self.test_penalty * n_tests[day]

        return rew
