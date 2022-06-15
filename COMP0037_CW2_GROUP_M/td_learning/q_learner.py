'''
Created on 19 Mar 2022

@author: ucacsjj
'''

import random

from airport.driving_actions import DrivingActionType

from .td_learner_base import TDLearnerBase

class QLearner(TDLearnerBase):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDLearnerBase.__init__(self, environment)
        
        self._q = None
        self.returns = [] # list to hold the rewards from all episodes

    def initialize(self, q):
        self._q = q
        
        if self._q.policy() is not None:
            self._q.policy().set_epsilon(self._epsilon)

    def _learn_online_from_episode(self):
        
        # Initialize a random state
        S = self._environment.pick_random_start()
        assert(S is not None)
        self._environment.reset(S)
        returns = 0
        
        # Main loop
        done = False
        
        while done is False:
            # Sample the action
            A = self._q.policy().sample_action(S[0], S[1])
           
            # Step the environment
            S_prime, R, done, info = self._environment.step(A)
            returns += R
            # Q3b : Replace with code to implement Q-learning
            # check if we have reached a terminal state, because then S_prime is None
            if done:
                break

            # find max value of q
            max_q = float('-inf')
            for a in range(10):
                action = DrivingActionType(a)
                dummy = self._q.value(S_prime[0], S_prime[1], action)
                if dummy > max_q:
                    max_q = dummy

            # implement the equation for updating self._q
            new_q = self._q.value(S[0], S[1], A) + self.alpha() * (R + self.gamma() * max_q - self._q.value(S[0], S[1], A))

            # update self._q
            self._q.set_value(S[0], S[1], A, new_q)
           
            # Store the state                
            S = S_prime

        # save the reward from the current episode
        self.returns.append(returns)
        