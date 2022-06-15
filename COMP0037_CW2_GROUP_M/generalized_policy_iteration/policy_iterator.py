'''
Created on 29 Jan 2022

@author: ucacsjj
'''

# This class implements the policy iterator algorithm.

from .dynamic_programming_base import DynamicProgrammingBase

class PolicyIterator(DynamicProgrammingBase):

    def __init__(self, environment):
        DynamicProgrammingBase.__init__(self, environment)
        
        # The maximum number of times the policy evaluation algorithm
        # will be run before the for loop is exited.
        self._max_policy_evaluation_steps_per_iteration = 100
        
        
        # The maximum number of times the policy evaluation iteration
        # is carried out.
        self._max_policy_iteration_steps = 1000
        

    def solve_policy(self):
                            
        # Initialize the drawers if defined
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()

        # Reset termination indicators       
        policy_iteration_step = 0        
        policy_stable = False
        
        # Loop until either the policy converges or we ran out of steps        
        while (policy_stable is False) and \
            (policy_iteration_step < self._max_policy_iteration_steps):
            
            # Evaluate the policy
            self._evaluate_policy()

            # Improve the policy            
            policy_stable = self._improve_policy()
            
            # Update the drawers if needed
            if self._policy_drawer is not None:
                self._policy_drawer.update()
                
            if self._value_drawer is not None:
                self._value_drawer.update()
                
            policy_iteration_step += 1

        # Draw one last time to clear any transients which might
        # draw changes
        if self._policy_drawer is not None:
            self._policy_drawer.update()
            
        if self._value_drawer is not None:
            self._value_drawer.update()

        # Return the value function and policy of the solution
        return self._v, self._pi

        
    def _evaluate_policy(self):
        
        # Get the environment and map
        environment = self._environment
        map = environment.map()
        
        # Execute the loop at least once
        
        iteration = 0
        
        while True:
            
            delta = 0

            # Sweep systematically over all the states            
            for x in range(map.width()):
                for y in range(map.height()):
                    
                    # We skip obstructions and terminals. If a cell is obstructed,
                    # there's no action the robot can take to access it, so it doesn't
                    # count. If the cell is terminal, it executes the terminal action
                    # state. The value of the value of the terminal cell is the reward.
                    # The reward itself was set up as part of the initial conditions for the
                    # value function.
                    if map.cell(x, y).is_obstruction() or map.cell(x, y).is_terminal():
                        continue
                                       
                    # Unfortunately the need to use coordinates is a bit inefficient, due
                    # to legacy code
                    cell = (x, y)
                    
                    # Get the previous value function
                    old_v = self._v.value(x, y)

                    # Compute p(s',r|s,a)
                    s_prime, r, p = environment.next_state_and_reward_distribution(cell, \
                                                                                     self._pi.action(x, y))
                    
                    # Sum over the rewards
                    new_v = 0
                    for t in range(len(p)):
                        sc = s_prime[t].coords()
                        new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))                        
                        
                    # Set the new value in the value function
                    self._v.set_value(x, y, new_v)
                                        
                    # Update the maximum deviation
                    delta = max(delta, abs(old_v-new_v))
 
            # Increment the policy evaluation counter        
            iteration += 1
                       
            print(f'Finished policy evaluation iteration {iteration}')
            
            # Terminate the loop if either the change was very small, or we exceeded
            # the maximum number of iterations.
            if (delta < self._theta) or (iteration >= self._max_policy_evaluation_steps_per_iteration):
                    break

    def _improve_policy(self):
       
        # Get the environment and map
        environment = self._environment
        map = environment.map()
        
        policy_stable = True
                
        # Sweep systematically over all the states            
        for x in range(map.width()):
            for y in range(map.height()):
                
                # The actions for both the terminal states and obstructions are fixed
                if (map.cell(x,y).is_terminal() is True) or \
                    (map.cell(x,y).is_obstruction() is True):
                    continue
                
                # Store the old action
                old_action = self._pi.action(x, y)
                
                cell = (x, y)
                
                # For computing the maximum value
                max_return = -float('inf')
                max_action = -1
                
                # Get the actions available at the current state
                available_actions = environment.available_actions()
                
                for a in range(0, available_actions.n):
                        
                    s_prime, r, p = environment.next_state_and_reward_distribution(cell, a)
                   
                    # Sum up over the reward / state distribution.
                    new_v=0
                    for t in range(len(p)):
                        sc = s_prime[t].coords()
                        new_v = new_v + p[t] * (r[t] + self._gamma * self._v.value(sc[0], sc[1]))
                
                    # If the reward is higher, flag this as the new best reward
                    if new_v > max_return:
                        max_return = new_v
                        max_action = a

                    # Sanity check                       
                    assert max_action != -1

                # If we changed the policy, then flag that the policy is not
                # stable and assign the new action 
                if max_action != old_action:
                    policy_stable = False
                    self._pi.set_action(x, y, max_action)
                    
        return policy_stable
                    
                
                
                
            