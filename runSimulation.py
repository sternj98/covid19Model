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
# create a population under some parameters and run a simulation of infection over time
# note that one timestep is a day
nDays = 90
size = 2000 # population of guilford (2016)
I0 = 5
p_connect = 4 / size # have connections with avg of 10 people

population = Population(size,age_dist,I0,p_connect)

population.prepSimulation(nDays)
p_infect = .05 # get this from diamond cruise ship

population.showConnections()
plt.show() # this just takes a lot of time in large population

widgets = [progressbar.Percentage(), progressbar.Bar()]
bar = progressbar.ProgressBar(widgets=widgets,maxval=nDays).start()
for day in range(1,nDays+1): # count 0 as initial day
    bar.update(day)

    # 1. propagate infection and advance infections
    nInfected = len(population.infectedPool)
    infectCount = 0



    # draw a random vector of # interax of len(infectedPool) w/ Pois or Binomial noise
    for infected in population.infectedPool: # use iter to get the index
        infectCount += 1 # then we can also get rid of the infectCount
        new_status = infected.step()
        if new_status != INF: # change status
            population.infectedPool.remove(infected) # take out of infected pool
            population.statuses[infected.id] = new_status
        else: # we're still infected
            for conn_idx in infected.connections:
                if population.people[conn_idx].status == SUS:
                    if rnd.rand() < p_infect:
                        population.people[conn_idx].getInfected()
                        population.infectedPool.append(population.people[conn_idx])
                        population.statuses[conn_idx] = INF
        if infectCount == nInfected: # to prevent infinite infection recursion
            break

    # 2. record the new population statistics
    population.nSus[day] = len(np.where(population.statuses == SUS)[0])
    population.nRec[day] = len(np.where(population.statuses == REC)[0])
    population.nInf[day] = len(np.where(population.statuses == INF)[0])
    population.nDead[day] = len(np.where(population.statuses == DEAD)[0])

    # 3. test (demo of dynamic testing)
    if day < 30:
        n_tests = 50
    else:
        n_tests = 500
    population.test(n_tests,day)

plt.figure()

# should really get a line in here to dynamically reshape lol
sns.heatmap(population.statuses.reshape((50,40)),cbar = False)
plt.title("Heatmap of Individual Outcomes for Average %i Interactions"%(int(size * p_connect)))
# plt.title("'\"Opening the Country Up\": Heatmap of Individual Outcomes for Average 3->10 Interactions")
population.plotStatistics(testing = True)
print("Dead:",population.nDead[-1])
plt.show()
