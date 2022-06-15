#!/usr/bin/env python3

'''
Created on 29 Jan 2022

@author: ucacsjj
'''


from airport.scenarios import *
from airport.airport_driving_environment import AirportDrivingEnvironment
from airport.driving_actions import DrivingActionType

from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_iterator import ValueIterator

from airport.driving_policy_drawer import DrivingPolicyDrawer

# This script illustrates how to use the airport environment

if __name__ == '__main__':

    
    # Get the map for the scenario
    airport_map, drawer_height = three_row_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = AirportDrivingEnvironment(airport_map)
    
    # Create a simple environment
    policy_iterator = PolicyIterator(airport_environment)
    
    policy_iterator.initialize()
    
    policy_drawer = DrivingPolicyDrawer(policy_iterator.policy(), drawer_height)
    
    policy_iterator.set_policy_drawer(policy_drawer)
    
    v, pi = policy_iterator.solve_policy()
    
    
    policy_drawer.save_screenshot("policy_iteration_results.jpg")
    
    pi.show()
    
    
    
    policy_drawer.update()
    
    policy_drawer.wait_for_key_press()
    
    
    value_iterator = ValueIterator(airport_environment)
    
    value_iterator.initialize()
     
    policy_drawer = DrivingPolicyDrawer(value_iterator.policy(), drawer_height)
    
    value_iterator.set_policy_drawer(policy_drawer)
   
    #while True:
   
    for c in range(10):
        v, pi = value_iterator.solve_policy()
        policy_drawer.update()
    
    policy_drawer.save_screenshot("value_iteration_results.jpg")
    
    policy_drawer.wait_for_key_press()


