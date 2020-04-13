import numpy as np
from numpy import random as rnd
import random
import seaborn as sns
from matplotlib import pyplot as plt

# defining constant integers to make labeling easy
SUS = 0 # susceptible
INF = 1 # infected
REC = 2 # recovered
DEAD = 3 # dead

incubation = 5 # 5 days of infecting people, then isolate

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

age_keys = list(deathrate.keys())

class Person:
    def __init__(self,age,id):
        self.id = id
        self.age = age # dictionary key in {0,20,45,55,65,75,85}
        self.dr = deathrate[age]
        self.status = SUS # start as healthy susceptible
        self.connections = []

    # Define connected individuals to potentially infect later
    def defineConnections(self,connVector):
        for citizen in range(len(connVector)):
            if connVector[citizen] > 0:
                self.connections.append(citizen) # this is a person we may infect

    def getInfected(self):
        self.status = INF
        self.infection_counter = 0 # days sick
        # query mortality: (incr deathrate now for testing)
        if rnd.rand() <  deathrate[self.age]:
            self.fatal = True   # should be 21
            self.days_to_death = round(rnd.normal(10,3)) # people stay in the hospital for a long time
        else:
            self.fatal = False
            self.days_to_recovery = round(rnd.normal(10,7)) # probably around 3 weeks with a large std

    def step(self):
        self.infection_counter += 1 # add a day to the infection counter
        if self.fatal == True:
            if self.infection_counter == self.days_to_death:
                self.status = DEAD
        else:
            if self.infection_counter == self.days_to_recovery:
                self.status = REC
        if self.infection_counter > incubation:
            self.connections = [] # quarantine; don't come out because don't care about recovered

        return self.status

class Population:
    def __init__(self,size,age_dist,I0,p_connect):
        self.size = size
        self.connections = np.zeros((self.size,self.size))
        self.I0 = I0

        self.statuses = np.zeros(self.size) # vector of statuses for visualization

        self.infectedPool = [] # initialize empty list that will be populated with infected

        self.people = [] # Define population in a list

        id = 0
        for agegroup in range(len(age_dist)):
            for citizens in range(int(size * age_dist[agegroup])):
                self.people.append(Person(age_keys[agegroup],id))
                id += 1
        while len(self.people) < size:
            self.people.append(Person(age_keys[0],id)) # round it off with some babies
            id += 1
        # sparse, (currently) uniform, and static, connectivity matrix
        self.C = rnd.binomial(1,p_connect,size ** 2).reshape((size,size))
        np.fill_diagonal(self.C,0) # can't infect yourself you wanker

        for citizen in range(size):
            self.people[citizen].defineConnections(self.C[citizen,:])

        # infect initial people
        for i in range(I0):
            idx = random.choice(list(range(size)))
            self.people[idx].getInfected()
            self.statuses[i] = INF
            self.infectedPool.append(self.people[idx])

    def prepSimulation(self,nDays):
        """
            Initialize population to prepare for simulation of nDays
        """
        self.nSus = np.zeros(nDays+1)
        self.nSus[0] = self.size # initial susceptible
        self.nInf = np.zeros(nDays+1)
        self.nInf[0] = self.I0 # initial number of infected in population
        self.nRec = np.zeros(nDays+1)
        self.nRec[0] = 0
        self.nDead = np.zeros(nDays+1)
        self.nDead[0] = 0

    def showConnections(self):
        plt.figure()
        sns.heatmap(self.C,cmap='hot')
        plt.suptitle('Population Connectivity')

    def plotStatistics(self):
        plt.figure()
        plt.plot(self.nSus,label = "Susceptible")
        plt.plot(self.nInf,label = "Infected")
        plt.plot(self.nRec,label = "Recovered")
        plt.plot(self.nDead,label = "Dead")
        plt.legend()
        plt.title('Population Statistics over Time')
