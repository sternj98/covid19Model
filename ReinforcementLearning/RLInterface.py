import numpy as np
from matplotlib import pyplot as plt
import progressbar

# bring together agent and environment and run trials

class DeepQInterface():
    def __init__(self,agent,environment):
        self.agent = agent
        self.env = environment
        self.rewlist = []
        self.batch_size = 50 # sample 50 experiences when we update

    def step(self): # same process as above
        state = self.env.state.copy()
        action = self.agent.select_action(state) # agent selects action
        rew = self.env.execute_action(action) # execute agent action into the environment
        new_state = self.env.state.copy()
        if not np.all(self.env.state == self.env.terminal_state): # don't add terminal states to replay buffer
            self.agent.replay_buffer.push(state,action,new_state,rew)

        loss = self.agent.update(self.batch_size)
        self.losslist.append(loss) # append loss to assess performance over time
        return state,action,rew,new_state

    def runTrials(self,nTrials,assessmentInterval):
        counter = 0 # for batch training
        self.rewlist = []
        self.losslist = []
        self.eps = []

        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets,maxval=nTrials).start()

        for i in range(nTrials):
            bar.update(i)
            self.env.reset()
            total_rew = 0
            tstates,tactions,trews,tnewstates = [] , [] , [] , [] # accumulate states to debug
            while not self.env.episode_complete: # while the game is not over, keep taking actions
                state,action,rew,new_state = self.step()
                total_rew += rew
                # print(action,"\t",rew,"\t")

                tstates.append(state)
                tactions.append(action)
                trews.append(rew)
                tnewstates.append(tnewstates)
                counter += 1

            if i % assessmentInterval == 0:
                plt.plot(self.env.population.nInf,color = [i/nTrials,0,1 - i/nTrials])
            self.rewlist.append(total_rew)

            if counter % self.agent.target_update == 0: # update the target network
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())
            # update agent epsilon
            self.agent.epsilon = (self.agent.eps_end + (self.agent.eps_start - self.agent.eps_end) *
                                                        np.exp(-1. * counter / self.agent.eps_decay))
            self.eps.append(self.agent.epsilon)
