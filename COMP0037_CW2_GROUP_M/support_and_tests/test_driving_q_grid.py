#!/usr/bin/env python3

'''
Created on 21 Mar 2022

@author: ucacsjj
'''

from airport.scenarios import *
from airport.airport_driving_environment import AirportDrivingEnvironment
from airport.airport_map_drawer import AirportMapDrawer
from airport.driving_policy_drawer import DrivingPolicyDrawer
from airport.driving_actions import DrivingActionType
from airport.driving_q_grid import DrivingQGrid

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from td_learning.sarsa_learner import SarsaLearner
from td_learning.q_learner import QLearner

if __name__ == '__main__':
    
        # Create test environment
    airport_map, drawer_height = one_row_scenario()#
    airport_map, drawer_height = corridor_scenario()
    #airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    
    #airport_map_drawer.update()
    
    airport = AirportDrivingEnvironment(airport_map)
    airport.set_nominal_direction_probability(1)
    
    q_grid = DrivingQGrid('qgrid', airport_map)
    q_grid.show()
    
    policy_drawer = DrivingPolicyDrawer(q_grid.policy(), drawer_height)
    
    policy_drawer.update()
    policy_drawer.wait_for_key_press()
    
    value_function_drawer = ValueFunctionDrawer(q_grid.value_function(), drawer_height)
    value_function_drawer.update()
    
    #learner = QLearner(airport)
    learner = SarsaLearner(airport)
    learner.initialize(q_grid)
    
    learner.set_gamma(1)
    learner.set_alpha(1e-3)
    learner.set_epsilon(0.05)
    
    for i in range(8000):
        learner.learn_policy()
        policy_drawer.update()
        value_function_drawer.update()
        #q_grid.show()
    policy_drawer.wait_for_key_press()
        
