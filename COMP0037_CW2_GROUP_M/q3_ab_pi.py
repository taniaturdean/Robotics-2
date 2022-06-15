#!/usr/bin/env python3

'''
Created on 23 Mar 2022

@author: ucacsjj
'''
from time import sleep

from airport.scenarios import *
from airport.airport_driving_environment import AirportDrivingEnvironment
from generalized_policy_iteration.policy_iterator import PolicyIterator
from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from airport.driving_policy_drawer import DrivingPolicyDrawer

if __name__ == '__main__':

    # Get the map for the scenario

    airport_map, drawer_height = test_3x3_scenario()
    
    # Set up the environment for the robot driving around
    airport_environment = AirportDrivingEnvironment(airport_map)
    
    # Configure the process model
    airport_environment.set_nominal_direction_probability(1.0)
    
    # Create the policy iterator
    policy_solver = PolicyIterator(airport_environment)
    
    # Set up initial state
    policy_solver.initialize()
        
    # Bind the drawer with the solver
    policy_drawer = DrivingPolicyDrawer(policy_solver.policy(), drawer_height)
    policy_solver.set_policy_drawer(policy_drawer)
    
    value_function_drawer = ValueFunctionDrawer(policy_solver.value_function(), drawer_height)
    policy_solver.set_value_function_drawer(value_function_drawer)
        
    # Compute the solution
    v, pi = policy_solver.solve_policy()
    
    # Wait for a key press
    sleep(2000)
    value_function_drawer.wait_for_key_press()
