import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from CovidRL_Agents import DeepQAgent
from RLInterface import DeepQInterface
from CovidRL_Environment import DiseaseEnvironment

# create a population under some parameters and run a simulation of infection over time
# note that one timestep is a day
nDays = 90
size = 2000 # population of guilford (2016)
I0 = 5
p_connect = 10 / size # have connections with avg of 10 people
p_infect = .05 # get this from diamond cruise ship
n_tests = 20

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

environment = DiseaseEnvironment(nDays,size,I0,p_connect,age_dist,p_infect,n_tests)

agent = DeepQAgent()

rl = DeepQInterface(agent,environment)

nTrials = 1000
assessmentInterval = 10
rl.runTrials(nTrials,assessmentInterval)


# Rews might wanna be normalized here
# need metrics here
# should really get a line in here to dynamically reshape lol
# sns.heatmap(population.statuses.reshape((50,40)),cbar = False)
# plt.title("Heatmap of Individual Outcomes for Average %i Interactions"%(int(size * p_connect)))
# # plt.title("'\"Opening the Country Up\": Heatmap of Individual Outcomes for Average 3->10 Interactions")
# population.plotStatistics(testing = True)
# print("Dead:",population.nDead[-1])
plt.figure()
plt.plot(rl.losslist)
plt.title("Loss over training")
plt.figure()
plt.plot(rl.rewlist)
plt.title("Cumulative episode reward over training")

plt.show()
