'''
Created on 18 Mar 2022

@author: ucacsjj
'''

import random

class TDLearnerBase(object):

    def __init__(self, environment):

        # The environment the system works with        
        self._environment = environment

        # The discount factor        
        self._gamma = 1
        self._epsilon = 0
        self._alpha = 1e-2
                
        # Flag to show if initialized
        self._initialized = False
        
        self._number_of_episodes = 100
        
        # Working scratch variables for the current value function
        self._q = None

        # Shows debug output interactively
        self._policy_drawer = None
        self._value_drawer = None
        
        
    def learn_online_policy(self):
        for episode_count in range(self._number_of_episodes):
            self._learn_online_from_episode()
            
        if self._policy_drawer is not None:
            self._policy_drawer.update()
                        
        if self._value_drawer is not None:
            self._value_drawer.update()


    def set_alpha(self, alpha):
        self._alpha = alpha
        
    def alpha(self):
        return self._alpha

    # Set the discount factor        
    def set_gamma(self, gamma):
        self._gamma = gamma

    # Retrieve the discount factor        
    def gamma(self):
        return self._gamma
    
    def set_epsilon(self, epsilon):
        self._epsilon = epsilon
        
        if self._q is not None:
            if self._q.policy() is not None:
                self._q.policy().set_epsilon(self._epsilon)
        
    def epsilon(self):
        return self._epsilon
    
    # Initialize the policy and value function. Must be called
    # before trying to solve for the policy.
    def initialize(self, initial_q):
        self._q = initial_q
        
    # Set the drawer which will show the policy.
    # If set, this will update interactively.
    def set_policy_drawer(self, policy_drawer):
        self._policy_drawer = policy_drawer
                          
    # Set the drawer which will show the value function.
    # If set, this will update interactively.                                    
    def set_value_function_drawer(self, value_drawer):
        self._value_drawer = value_drawer
  