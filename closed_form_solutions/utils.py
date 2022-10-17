import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt


#
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

def plot_values(V):
    # reshape the state-value function
    V = np.reshape(V, (4,12))
    # plot the state-value function
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(V, cmap='cool')
    for (j,i),label in np.ndenumerate(V):
        ax.text(i, j, np.round(label,3), ha='center', va='center', fontsize=14)
    plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.title('State-Value Function')
    plt.show()


#
class Solver():

    #
    def __init__(self, env):

        #
        self.env = env

        #
        self.Q = []

        #
        self.initialize_Q_table()

    #
    def initialize_Q_table(self):

        #
        # make dicitionary which holds for each state a tuple:
        #   [state, direction, value]
        self.Q = np.zeros((self.env.nS,
                           self.env.nA   # this holds the direction and the value of each direction
                           ))+np.nan

    #
    def get_inverse_direction(self, direction):

        return (direction+2)%4

    #
    def build_Q_value_propagation(self, verbose=False):

        # walk backwards from discvoered connecting edges building a Q value function for each as a (direction, value)

        default_array = np.zeros((self.env.nA),'float32')+np.nan

        # assume only a single final state for now
        final_state = self.final_states

        #
        self.Q[final_state] = self.final_rewards
        #
        # grab neighbours of current state
        current_state = final_state
        neighbours_and_dirs = self.near_states_and_directions[current_state]
        neighbours = neighbours_and_dirs.T[:,1]

        if verbose:
            print ("setting final state: ", final_state, " Q: ", self.Q[final_state])
            print ("neighbours and dirs: ", neighbours_and_dirs)
            print ("neighbours of ",current_state, " : ", neighbours)

        #
        states = neighbours

        #
        states_visited = []
        # loop over each available state and look for non-connected neighbours
        ctr2= 0
        while states.shape[0]>0:

            # grab random state
            # TODO can write even more efficient alg here that selects only locs from
            # - a sepcific list of "frontier" states
            # - until each frontier state has all actions exhausted
            state = states[0]

            #print ("selcte")
            # grab its neighbours and directions
            neighbours = self.near_states_and_directions[state][1]
            directions = self.near_states_and_directions[state][0]
            if verbose:
                print ("state selected: ", state, ", neigbhours: ", neighbours, ", directoins: ", directions)

            # loop over neighbours and connect to the one with maximum non-nan value
            max_vals = default_array.copy()
            for ctr,neighbour in enumerate(neighbours):

                # check all neighbours and connect to lowest value one
                # TODO a better way to search for all nans
                try:
                    max_vals[ctr] = np.nanmax(self.Q[neighbour])
                except:
                    max_vals[ctr] = np.nan

            ################ Check if there are any non-nan connetions #############
            #max_vals = np.array(max_vals,'float32')
            #idx = np.where(np.isnan(max_vals))[0]

            # check to see if we have any non-nans
            # TODO: try-catch is a bad idea here; use better logic
            try:
                max_argmax_idx = np.nanargmax(max_vals)
                if verbose:
                    print ("max vals: ", max_vals)
            except:
                # no connections found - put the state tothe back of the states
                states = np.roll(states,-1)
                #print (state, " no connection found ")
                continue

            ############## Grab the ID of the max connected by QValue
            neighbour_selected = neighbours[max_argmax_idx]
            #print ("found connecting state ... connecting ", state, " to ", neighbours[max_argmax_idx])

            # find the first non-nan value and connet to it
            #inverse_direction = self.get_inverse_direction(directions[max_argmax_idx])
            direction = directions[max_argmax_idx]
            #print ("direction: ", directions[max_argmax_idx])
            #       "ineverse direction: ", inverse_direction)

            # reset enviorment
            # TODO: not sure if this is required/ideal
            self.env.reset()

            # set agent to neighbour location
            self.env.set_state(state)

            # step agent from current state to the neighbour
            res = self.env.step(direction)

            # sanity check that back to the current state
            #print ("current state: ", state, ", action result: ", res, ", neighbour Q: ", self.Q[neighbour])

            # add reward to the Q[state] and the maximum possible value from the otherstate
            term1 = res[1]
            term2 = np.nanmax(self.Q[neighbour_selected])
            #print ("term1: ", term1, ", term2: ", term2)
            self.Q[state,direction] = term1 + term2
            #print ("updated state: ", state)

            # delete the current state from the possible states that can be chosen
            neighbours_and_dirs2 = self.near_states_and_directions[state]
            neighbours2 = neighbours_and_dirs2.T[:,1]

            #
            states = np.hstack((neighbours2, states))
            _, idx = np.unique(states, return_index=True)
            states = (states[np.sort(idx)])

            # add current visited state to the list of excluded states
            states_visited.append(state)

            # remove any previously visited states from the list
            mask_idx = np.in1d(states, states_visited)
            states = states[~mask_idx]

            #
            idx = np.random.choice(np.arange(states.shape[0]), states.shape[0], replace=False)
            states = states[idx]
            if verbose:
                print ("new states: ", states)
                print ("##########################")
            #break

            ctr2+=1
        #
        print ("Done in # steps: ", ctr2)

    #
    def correct_Q_table(self, verbose=False):

        '''  Loop over table N-states times and for each state check if there's a lower path nearby

        '''

        #
        for state0 in range(self.env.nS):
            n_changes = 0
            for state in range(self.env.nS):
                if verbose:
                    print (state, self.Q[state])

                if state in self.final_states:
                    continue

                neighbour_states = self.near_states_and_directions[state].T[:,1]
                neighbour_directions = self.near_states_and_directions[state].T[:,0]

                #
                max_vals_neighbours = []
                for neighbour_state in neighbour_states:
                    max_vals_neighbours.append(np.nanmax(self.Q[neighbour_state]))

                idx = np.nanargmax(max_vals_neighbours)

                # reset system
                # TODO: is this necessary?!
                if False:
                    self.env.reset()

                #
                if verbose:
                    print ("resetting state to ", neighbour_states[idx])
                self.env.set_state(neighbour_states[idx])

                #
                inverse_dir = self.get_inverse_direction(neighbour_directions[idx])
                res = self.env.step(inverse_dir)
                reward = res[1]

                #
                term1 = max_vals_neighbours[idx]
                term2 = reward
                term3 = term1 + term2

                if verbose:
                    print ("term1: ", term1, ", reward into current state: ", term2)
                    print ("")

                if np.nanmax(self.Q[state]) != term3:
                    self.Q[state] = term3
                    n_changes+=1
            if n_changes==0:
                print (" converted in # steps ", state0)
                break

    #
    def find_neighbour_states(self):

        ''' Function finds the nearby allowed states + direction

        '''


        #
        self.near_states_and_directions = []
        for k in range(self.env.nS):

            #
            neighbour_states =[]
            for a in range(self.env.nA):
                self.env.reset()
                self.env.set_state(k)

                #
                res = self.env.step(a)

                #
                neighbour_states.append([a, res[0]])

            # find unique states
            neighbour_states = np.array(neighbour_states)

            # remove self from list
            idx = np.where(neighbour_states[:,1]==k)[0]

            if idx.shape[0]>0:
                neighbour_states = np.delete(neighbour_states,idx,axis=0)

            #
            self.near_states_and_directions.append(neighbour_states.T)

    #def


    def find_final_states(self):

        #
        final_states = []
        final_rewards = []
        for k in range(self.env.nS):

            #
            for a in range(self.env.nA):
                self.env.reset()
                self.env.set_state(k)

                #
                res = self.env.step(a)

                if res[2]==True:

                    #
                    #print (res)

                    final_states.append(res[0])
                    final_rewards.append(res[1])

        res = np.unique(final_states, return_index=True)

        #
        self.final_states = res[0].squeeze()

        #
        idx = res[1].squeeze()
        self.final_rewards = final_rewards[idx]

    #
    def solve_with_do_operator(self):

        print ("solving with do operator")

        # find terminal state


        # walk backwards from terminal and make value table

        # iterate and update value table


    #
    def test(self):
        pass


















