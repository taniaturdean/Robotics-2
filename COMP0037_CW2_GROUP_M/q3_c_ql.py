#!/usr/bin/env python3

'''
Created on 21 Mar 2022

@author: ucacsjj
'''

import math

import numpy as np
from matplotlib import pyplot as plt

from airport.scenarios import *
from airport.airport_driving_environment import AirportDrivingEnvironment
from airport.airport_map_drawer import AirportMapDrawer
from airport.driving_policy_drawer import DrivingPolicyDrawer
from airport.driving_actions import DrivingActionType
from airport.driving_q_grid import DrivingQGrid

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from td_learning.q_learner import QLearner

if __name__ == '__main__':
    
    # Create test environment
    airport_map, drawer_height = corridor_scenario()
    airport = AirportDrivingEnvironment(airport_map)
    airport.set_nominal_direction_probability(0.8)
    
    q_grid = DrivingQGrid('Q Learner', airport_map)
    q_grid.show()
    
    learner = QLearner(airport)    
    learner.initialize(q_grid)
    learner.set_gamma(1)
    learner.set_alpha(1e-3)
    learner.set_epsilon(1)
            
    # Bind the drawer with the solver
    policy_drawer = DrivingPolicyDrawer(q_grid.policy(), drawer_height)
    learner.set_policy_drawer(policy_drawer)
    
    value_function_drawer = ValueFunctionDrawer(q_grid.value_function(), drawer_height)
    learner.set_value_function_drawer(value_function_drawer)
    
    # Run the learning algorithm.
    for i in range(100000):
        print(i)
        learner.set_epsilon(1/math.sqrt(math.sqrt(i+1)))
        learner.learn_online_policy()
        q_grid.show()

    x_avg = []
    y_avg = []
    total = 0
    length = len(learner.returns)
    for i in range(length):
        x_avg.append(i)
        total += learner.returns[i]
        y_avg.append(total / (i + 1))

    x_avg = np.array(x_avg)
    y_avg = np.array(y_avg)
    plt.title("Q-learning Cumulative Averaged")
    plt.xlabel("Num of episodes")
    plt.ylabel("Rewards")
    plt.plot(x_avg, y_avg)
    plt.show()

    x = np.array(range(0, 10000000))
    y = np.array(learner.returns)
    plt.title("Q-learning Rewards")
    plt.xlabel("Num of episodes")
    plt.ylabel("Rewards")
    plt.plot(x, y)
    plt.show()

    policy_drawer.wait_for_key_press()
        
