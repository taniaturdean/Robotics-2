'''
Created on 29 Jan 2022

@author: ucacsjj
'''

# This grid should be used to store the policy at each cell.
# The action is very system dependent, and so we can't provide
# any more details at this level in the class

from random import random

from grid_search.grid import Grid

class PolicyGrid(Grid):

    def __init__(self, name, environment_map, set_random = False):
        Grid.__init__(self, name, \
                      environment_map.width(), environment_map.height())
        
        self._epsilon = 0.01
        self._environment_map = environment_map
    
    def environment_map(self):
        return self._environment_map
    
    def set_action(self, x, y, action):
        raise NotImplementedError()
        
    def action(self, x, y):
        raise NotImplementedError()
    
    def set_epsilon(self, epsilon):
        self._epsilon = epsilon
    
    def sample_action(self, x, y):
        
        p = random()
    
        # Sample using e-greedy
        if p < self._epsilon:
            #print('Random')
            return self._draw_random_action(x, y)
        else:
            #print('Not random')

            return self.action(x, y)
        
        
    # Sample a random action
    def _draw_random_action(self, x, y):
        raise NotImplementedError()
        
       