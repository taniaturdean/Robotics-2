'''
Created on 21 Mar 2022

@author: ucacsjj
'''

import math

from generalized_policy_iteration.q_grid import QGrid
from generalized_policy_iteration.value_grid import ValueGrid
from .driving_policy import DrivingPolicy
from .driving_actions import DrivingActionType

class DrivingQGrid(QGrid):

    def __init__(self, name, environment_map, set_random = False):
        
        # Allocate the Q grid with the correct number of actions
        QGrid.__init__(self, name, environment_map, DrivingActionType.NUMBER_OF_ACTIONS, \
                       set_random)
        
                
        # Create the value function
        self._v = ValueGrid(name + ": Value Function", environment_map)
        
        # Assign the Q values. From https://stackoverflow.com/questions/367565/how-do-i-build-a-numpy-array-from-a-generator
        # there is no simple generator to use, so for loops it is! The main thing we do is:
        # Assign -inf on terminal states to all actions apart from TERMINATE; for TERMINATE assign the terminal reward
        # Assign -inf on obstructed states for all actions apart from NONE; for NONE assign 0
        # All others are set to zero.
        
        for x in range(self._width):
            for y in range(self._height):
                
                is_terminal = environment_map.cell(x, y).is_terminal()
                is_obstruction = environment_map.cell(x, y).is_obstruction()
                         
                if is_terminal is True:
                    for a in range(DrivingActionType.NUMBER_OF_ACTIONS):
                        self._q_values[x, y, a] = -float('inf')
                    terminate_reward = environment_map.cell(x, y).params()
                    self._q_values[x, y, DrivingActionType.TERMINATE] = \
                        terminate_reward
                    self._v.set_value(x, y, terminate_reward)
                elif is_obstruction is True:
                    for a in range(DrivingActionType.NUMBER_OF_ACTIONS):
                        self._q_values[x, y, a] = -float('inf')
                        self._q_values[x, y, DrivingActionType.NONE] = 0
                    self._v.set_value(x, y, math.nan)
                else:
                    self._q_values[x, y, DrivingActionType.TERMINATE] = -float('inf')
       
       
        print(self._q_values)
       
        # Create the policy; this assigns initial values as well based off of the map
        self._policy = DrivingPolicy(name + ": Policy", environment_map)
