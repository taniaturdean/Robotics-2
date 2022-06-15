'''
Created on 19 Mar 2022

@author: ucacsjj
'''

import random

from .td_learner_base import TDLearnerBase

class SarsaLearner(TDLearnerBase):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDLearnerBase.__init__(self, environment)
        self.returns = [] # list to hold the rewards from all episodes

    def initialize(self, q):
        self._q = q

    def _learn_online_from_episode(self):
        
        # Initialize a random state
        S = self._environment.pick_random_start()
        assert(S is not None)
        self._environment.reset(S)
        returns = 0
                   
        # Pick the first action
        A = self._q.policy().sample_action(S[0], S[1])
         
        # Main loop
        done = False
           
        while done is False:
            S_prime, R, done, info = self._environment.step(A)
            # returns += R
            # Q3a: Replace with code to implement SARSA
            # check if we have reached a terminal state, because then S_prime is None
            if done:
                returns += R
                break
            # choose an action A_prime
            A_prime = self._q.policy().sample_action(S_prime[0], S_prime[1])
            # implement the equation for updating self._q
            new_q = self._q.value(S[0], S[1], A) + self.alpha() * (R + self.gamma() * self._q.value(S_prime[0], S_prime[1], A_prime) - self._q.value(S[0], S[1], A))

            # update self._q
            self._q.set_value(S[0], S[1], A, new_q)
   
            # Store the state                
            S = S_prime
            A = A_prime

        # save the reward from the current episode
        self.returns.append(returns)
