'''
Created on 29 Jan 2022

@author: ucacsjj
'''

import random

from .driving_actions import DrivingActionType

from generalized_policy_iteration.policy_grid import PolicyGrid

# The driving policy. For each cell in the state space, in which direction do we go next?

class DrivingPolicy(PolicyGrid):

    def __init__(self, name, airport_map):
        PolicyGrid.__init__(self, name, airport_map)
        
        # When we set up the policy, we MUST put a TERMINATE action in the cells which
        # are terminals. If the cell is in a wall, the action is flagged to NONE. For
        # all other cells, the initial strategy is completely random
        
        type_creator = lambda x, y: DrivingActionType.TERMINATE if airport_map.cell(x, y).is_terminal() \
                                    else DrivingActionType.NONE if airport_map.cell(x, y).is_obstruction() \
                                    else DrivingActionType(random.randrange(DrivingActionType.RIGHT, DrivingActionType.NONE + 1))
        
        self._policy = [[type_creator(x,y) for y in range(self._height)] \
                            for x in range(self._width)]

    def set_action(self, x, y, action):
                
        # Sanity check to ensure that everything is okay
        if self._environment_map.cell(x, y).is_terminal() is True:
            if int(action) is not int(DrivingActionType.TERMINATE):
                raise AssertionError('Can only set the action to TERMINATE on a terminal state.')            
        
        self._policy[x][y] = DrivingActionType(action)
        
    def action(self, x, y):
        if self._environment_map.cell(x, y).is_terminal() is True:
            assert(self._policy[x][y] is DrivingActionType.TERMINATE)
            
        return self._policy[x][y]
       
    def show(self):
        
        # Print out the policy as a string. Note we have to reverse y because
        # y=0 is at the origin and so we need to print top-to-bottom
        for y in reversed(range(self._height)):
            line_string = str(int(self._policy[0][y]))
            for x in range(1,self._width):
                line_string += str(" ") + str(int(self._policy[x][y]))
            print(line_string)
        
    def _draw_random_action(self, x, y):
        
        # If this is a terminal, the only action is terminate, otherwise sample randomly
        if self._environment_map.cell(x, y).is_terminal() is True:
            action = DrivingActionType.TERMINATE
        else:
            action = DrivingActionType(random.randrange(DrivingActionType.RIGHT, DrivingActionType.NONE + 1))
            #print(action)
            
        #print(f'A={str(DrivingActionType(action))}')
 
        return action
    