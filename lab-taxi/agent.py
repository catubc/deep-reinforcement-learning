import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self,
                 nA=6,
                 epsilon=0.75,
                 epsilon_scaling = 1.0001,
                 alpha=0.2,
                 gamma=1.0):

        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.epsilon = epsilon
        self.epsilon_scaling = epsilon_scaling
        self.alpha = alpha
        self.gamma = gamma

        #
        self.nA = nA

        #
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        #
        self.action_mask_t1 = np.ones(self.nA)

        #
        self.initalize_Q_function()

        #
        self.step = self.step_sarsa

    #
    def initalize_Q_function(self):

        #
        for k in range(500):
            self.Q[k]

    #
    def get_updated_epsilon(self):

        #
        temp = min(0.98, self.epsilon*((self.epsilon_scaling)**self.i_episode))

        return temp

    #
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        # get probabilities of the state
        probs = self.Q[state]
        probs = probs+abs(np.min(probs))

        # exclude non possible actions based on reward return from the world;
        probs = probs*self.action_mask_t1


        # if all actions are same, then split the probs
        if np.sum(probs)>0:
            action = np.random.choice(np.arange(self.nA), p=probs/np.sum(probs))
        else:
            action = np.random.choice(np.arange(self.nA))

        # for prediction step use greedy policy
        if self.greedy_policy==True:
            return action

        # for training implement epsilon stochasticity
        idx = np.random.rand()
        if idx<self.get_updated_epsilon():
            return action

        # otherwise just select a random action
        action = np.random.choice(np.arange(self.nA))

        #
        return action

    #
    def compute_expected_val(self, state):

        # grab probs and normalize so they start at zero; and also sum up to 1
        probs = self.Q[state]
        probs = (probs-np.min(probs))

        #
        if np.sum(probs)==0:
            probs = np.zeros(4)*0.25
        else:
            probs = probs/np.sum(probs)

       # print ("probs: ", probs)
       # print ("Q vals: ", Q[state_t1])

        #
        total_val = 0
        for k in range(4):
            prob = probs[k]
            val = self.Q[state][k]

            total_val += prob*val

        #
        return total_val

    #
    def step_Q_expected(self,
               state_t0,
               action_t0,
               reward_t1,
               #action_t1,
               state_t1,
               done
               #alpha,
               #gamma
              ):

       # exit if espisode complete
        if done:
            return

        # chose action at At+1 using epsilon greedy policy from Q
        action_t1 = self.select_action(state_t1)

        #
        term1 = self.Q[state_t0][action_t0]

        # here select the most likely action , rather than action t1
        # not super clear if we have to run this through epsilon-greedy discounter
        expected_val = self.compute_expected_val(state_t1)      # is this really the correct or is it the next


        # same as sarsa, but the action is the optimal one that would be taken
        term2 = self.gamma*expected_val - self.Q[state_t0][action_t0]

        #
        term3 = self.alpha*(reward_t1 + term2)

        #
        total = term1+term3

        #
        self.Q[state_t0][action_t0] = total

        #
        return action_t1

    #
    def step_sarsa(self,
                   state_t0,
                   action_t0,
                   reward_t1,
                   state_t1,
                   #action_t1,
                   done):

        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        # exit if espisode complete
        if done:
            return

        # chose action at At+1 using epsilon greedy policy from Q
        action_t1 = self.select_action(state_t1)

        # proposed dictionary action based on TD learning I guess
        # self.Q[state][action] += 1

        #
        term1 = self.Q[state_t0][action_t0]

        #
        term2 = self.gamma*self.Q[state_t1][action_t1] - self.Q[state_t0][action_t0]

        #
        term3 = self.alpha*(reward_t1 + term2)

        #
        total = term1+term3

        #
        self.Q[state_t0][action_t0] = total

        return action_t1

    #
    def get_action_epsilon_greedy(self):
        #, Q, state_t0, epsilon):

        #
        probs = Q[state_t0]
        probs = probs+abs(np.min(probs))
       # print ("probs: ", probs)

        #
        if np.sum(probs)>0:
            action = np.random.choice(np.arange(4), p=probs/np.sum(probs))
        else:
            action = np.random.choice(np.arange(4))

        #
        #print ("Q based action: ", action)

        #
        idx = np.random.rand()
        if idx<epsilon:
            return action

        # otherwise just select a random action
        action = np.random.choice(np.arange(4))

        #
        return action
