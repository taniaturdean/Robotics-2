'''
Created on 19 Mar 2022

@author: ucacsjj
'''

# This grid stores the value function for each state. It's defined to be a 
# real number in all cases, so we specialise it here. In addition, it
# automatically creates a policy the policy is a grid the same size as the array.
# The 

import random

import numpy as np

from grid_search.grid import Grid
from .policy_grid import PolicyGrid

class QGrid(Grid):
    '''
    classdocs
    '''

    def __init__(self, name, environment_map, num_actions, set_random = False):
        
        #print(name)
        #print(environment_map)
        #print(num_actions)
        
        Grid.__init__(self, name, \
                      environment_map.width(), environment_map.height())
        
        self._num_actions = num_actions
        self._policy = None
        self._v = None
        
        # Set random isn't used here; we assume a zero value is arbitrary enough
        self._q_values = np.zeros((self._width, self._height, num_actions))
    
    def policy(self):
        return self._policy
    
    def value_function(self):
        return self._v
    
    def num_actions(self):
        return self._num_actions    
        
    def set_value(self, x, y, a, Q):
        
        # Store the value
        self._q_values[x, y, a] = Q
        
        # Find the maximum value for the action at this cell, and set it for the policy
        if self._policy is not None and self._v is not None:
            action_values = self._q_values[x, y, :]
            #print(f'actions={actions}')
            
            # Find the max actions; note horrible subscripting
            max_actions = (np.where(action_values == np.amax(action_values)))[0]
            
            # Note that multiple actions might have the same q value. If that's the case,
            # pick one at random
            #print(random.choice(range(max_actions.size)))
            max_action = max_actions[random.choice(range(max_actions.size))]
            
            #print(f'max_action={max_action}')
            if self._policy is not None:
                self._policy.set_action(x, y, max_action)
            
            # Set the current value function
            if self._v is not None:
                self._v.set_value(x, y, action_values[max_action])
        
    def value(self, x, y, a):
        return self._q_values[x, y, a]
    
    def values_of_actions(self, x, y):
        return self._q_values[x, y, :]
    
    def show(self):
    
        # Print out the policy as a string. Note we have to reverse y because
        # y=0 is at the origin and so we need to print top-to-bottom
        for a in range(0, self._num_actions):
            print(f'Action={a}:')
            for y in reversed(range(self._height)):
                line_string = "{:.3f}".format(self._q_values[0,y,a])
                for x in range(1,self._width):
                    line_string += str(" ") + "{:.3f}".format(self._q_values[x,y,a])
                print(line_string)
            print('========================================================')
