import numpy as np
from numpy import random as rnd

# defining constant integers to make labeling easy
SUS = 0 # susceptible
INF = 1 # infected
REC = 2 # recovered
DEAD = 3 # dead

# vector of deathrates for age categories;
# data from https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm#F2_down
# age categories: 0-19,20-44,45-54,55-65,65-74,75-84,>85
deathrate = {0: 0,
             20:.0015,
             45: .0065,
             55: .02,
             65: .035,
             75: .075,
             85: .15}

p_infect = .9999 # get this from diamond cruise ship

class Person:
    def __init__(self,age):
        self.age = age # dictionary key in {0,20,45,55,65,75,85}
        self.dr = deathrate[age]
        self.status = SUS # start as healthy susceptible

    # these connections are constant, other connections are randomly chosed every day
    def defineFamily(self,connVector):
        self.connections = []
        for neuronIndex in range(len(outputWeightVector)):
            if outputWeightVector[neuronIndex] > 0:
                self.outputs.append([neuronIndex,outputWeightVector[neuronIndex],outputDelayVector[neuronIndex]])

    def getInfected(self):
        self.status = INF
        self.infection_counter = 0 # days sick
        # query mortality:
        if rnd.rand() < deathrate[self.age]:
            self.fatal = True
            self.days_to_death = round(rnd.normal(25,3)) # people stay in the hospital for a long time

class Population:
    def __init__(self,age_vector,I0):
        self.size = np.sum(age_vector)
        self.nSus = self.size
        self.nInf = I0 # initial number of infected in population
        self.nRec = 0
        self.nDead = 0
        self.connections = np.zeros((self.size,self.size))

    def defineFamilyConnections(self):
