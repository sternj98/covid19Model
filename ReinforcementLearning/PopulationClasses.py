import numpy as np
from numpy import random as rnd
import random
import seaborn as sns
from matplotlib import pyplot as plt
import progressbar
import pandas as pd

# defining constant integers to make labeling easy
SUS = 0 # susceptible
INF = 1 # infected
REC = 2 # recovered
DEAD = 3 # dead

incubation = 5 # 5 days of infecting people, then isolate

# vector of deathrates for age categories;
# data from https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm#F2_down
# age categories: 0-19,20-44,45-54,55-65,65-74,75-84,>85
# Upper end of distributions, justified by hostpital overcrowding effects
deathrate = {0: 0,
             20:.004,
             45: .015,
             55: .033,
             65: .055,
             75: .105,
             85: .25}

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
            self.fatal = True
            self.days_to_death = round(rnd.normal(21,3)) # people stay in the hospital for a long time
        else:
            self.fatal = False
            self.days_to_recovery = round(rnd.normal(14,7)) # probably around 3 weeks with a large std

    def step(self):
        self.infection_counter += 1 # add a day to the infection counter
        if self.fatal == True:
            if self.infection_counter == self.days_to_death:
                self.status = DEAD
        else:
            if self.infection_counter == self.days_to_recovery:
                self.status = REC
        if self.infection_counter > incubation: # put this on gaussian?
            self.connections = [] # quarantine; don't come out because don't care about recovered

        return self.status

class Population:
    def __init__(self,size,age_dist,I0,p_connect):
        self.size = size
        self.connections = np.zeros((self.size,self.size))
        self.I0 = I0
        self.p_connect = p_connect

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
        self.C = rnd.binomial(1,2 * p_connect,size ** 2).reshape((size,size)) # symmetric ?
        np.fill_diagonal(self.C,0) # can't infect yourself you wanker

        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets,maxval=size).start()
        counter = 0
        for citizen in range(size):
            counter += 1
            bar.update(counter)
            self.people[citizen].defineConnections(self.C[citizen,:])

        # infect initial people
        for i in range(I0):
            idx = random.choice(list(range(size)))
            self.people[idx].getInfected()
            self.statuses[i] = INF
            self.infectedPool.append(self.people[idx])

    def prepSimulation(self,nDays,p_infect):
        """
            Initialize population to prepare for simulation of nDays
        """
        self.p_infect = p_infect

        self.nSus = np.zeros(nDays+1)
        self.nSus[0] = self.size # initial susceptible
        self.nInf = np.zeros(nDays+1)
        self.nInf[0] = self.I0 # initial number of infected in population
        self.nRec = np.zeros(nDays+1)
        self.nRec[0] = 0
        self.nDead = np.zeros(nDays+1)
        self.nDead[0] = 0

        self.testInf = np.zeros(nDays + 1)
        self.testInf[0] = 0
        self.testHealthy = np.zeros(nDays + 1)
        self.testHealthy[0] = 0
        self.nTests = np.zeros(nDays + 1)
        self.nTests[0] = 1

    def test(self,n_tests,day):
        """
            Test a SRS of n_tests individuals and log # noninfected, # infected
            Currently assumes tests are perfect and sample is SRS
        """
        sample = random.sample(list(self.statuses),k = n_tests)
        self.testInf[day] = len(np.where(np.array(sample) == INF)[0])
        self.testHealthy[day] = n_tests - self.testInf[day]
        self.nTests[day] = n_tests

    def showConnections(self):
        plt.figure()
        sns.heatmap(self.C,cmap='hot')
        plt.suptitle('Population Connectivity')

    # would be nice to have this in seaborn
    def plotStatistics(self,testing = False):
        plt.figure()
        plt.plot(self.nSus,label = "Susceptible")
        plt.plot(self.nInf,label = "Infected")
        plt.plot(self.nRec,label = "Recovered")
        plt.plot(self.nDead,label = "Dead")
        if testing == True:
            # bernouilli squared error
            testEst = self.testInf * self.size / self.nTests
            stderror = self.size * np.sqrt((self.testInf / self.nTests) * (1 - self.testInf / self.nTests) / self.nTests)
            plt.plot(testEst)
            plt.errorbar(list(range(len(testEst))),testEst,stderror,label = "Infected estimate")


        plt.legend()
        plt.title('Population Statistics over Time for Average %i Interactions'%(int(self.size * self.p_connect)))

    def step(self,day,n_tests,policy):
        """
            Advance simulation by a step
        """
        # 1. propagate infection and advance infections
        nInfected = len(self.infectedPool)
        infectCount = 0

        # draw a random behavior with poisson noise
        # would be nice to have a noise model with variable variance
        behavior = rnd.poisson(policy, nInfected)

        for infected in self.infectedPool: # use iter to get the index
            infectCount += 1 # then we can also get rid of the infectCount
            new_status = infected.step()
            if new_status != INF: # change status
                self.infectedPool.remove(infected) # take out of infected pool
                self.statuses[infected.id] = new_status
            else: # we're still infected
                for conn_idx in infected.connections[:behavior[infectCount-1]]:
                    if self.people[conn_idx].status == SUS:
                        if rnd.rand() < self.p_infect:
                            self.people[conn_idx].getInfected()
                            self.infectedPool.append(self.people[conn_idx])
                            self.statuses[conn_idx] = INF
            if infectCount == nInfected: # to prevent infinite infection recursion
                break

        # 2. record the new population statistics
        self.nSus[day] = len(np.where(self.statuses == SUS)[0])
        self.nRec[day] = len(np.where(self.statuses == REC)[0])
        self.nInf[day] = len(np.where(self.statuses == INF)[0])
        self.nDead[day] = len(np.where(self.statuses == DEAD)[0])

        # 3. test (demo of dynamic testing)
        self.test(n_tests,day)
