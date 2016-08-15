import random
import sys

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import qlearn as qlearn

import pandas as pd 

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward', 'left', 'right']
        self.lastState = None
        self.lastAction = None
        self.lastReward = None

        self.reward_sum = 0 
        self.disc_reward_sum = 0 
        self.n_dest_reached = 0 
        self.last_dest_fail = 0 
        self.sum_time_left = 0 
        self.n_penalties = 0   
        self.last_penalty = 0 
        self.len_qvals = 0             
        
        self.ai = None

        if (sys.argv.count == 4):
            alpha = sys.argv[1]
            gamma = sys.argv[2]
            epsilon = sys.argv[3]
        else:
            alpha = 0.8
            gamma = 0.6
            epsilon = 0.0 

        self.ai = qlearn.QLearn(actions=self.actions, alpha=alpha, gamma=gamma, epsilon=epsilon)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        if (deadline == 0):
            self.last_dest_fail += 1

        # TODO: Update state

        # self.state = "{}".format(inputs) # agent reaches destination only ~ 1% of the trials
        # self.state = "{}-{}".format(inputs, deadline) # agent reaches destination only ~ 6% of the trials
        self.state = "{}-{}-{}".format(inputs, deadline, self.next_waypoint) # agent reaches destination only ~ 80% of the trials
        
        # TODO: Select action according to your policy

        # Q. 1 - agent with random actions
        # action = random.choice(self.actions)
        
        # Q. 2 - q-learning agent
        action = self.ai.chooseAction(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        # print "reward - {}".format(reward)
        if (reward < 0 ):
            self.n_penalties += 1
            self.last_penalty = t

        self.reward_sum += reward  
        self.disc_reward_sum += (1/(t+1)) * reward
        self.len_qvals = len(self.ai.q)             

        # TODO: Learn policy based on state, action, reward
        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, self.lastReward, self.state)

        self.lastState = self.state
        self.lastAction = action
        self.lastReward = reward

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    # sim = Simulator(e, update_delay=1.0, display=True)  # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    return sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    # print a.ai.q

if __name__ == '__main__':
    # run()
    results = []
    for i in range(100):
        sim_results = run()
        results.append(sim_results)
    df_results = pd.DataFrame(results)
    df_results.columns = ['reward_sum', 'disc_reward_sum', 'n_dest_reached',
                              'last_dest_fail', 'sum_time_left', 'n_penalties',
                              'last_penalty', 'len_qvals']
    # print df_results.describe()
    # df_results.to_csv('results.csv')
    df_results.describe().transpose().to_csv('results.csv', mode='a',sep='\t')