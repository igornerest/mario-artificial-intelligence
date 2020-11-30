

import sys
import os
import pickle
import retro
import time
from rominfo import *
from utils import *
import random
import itertools


def format_state(state):
    return ','.join(map(lambda x : str(int(x)), state))  

moves = {'corre':130, 'pula':131, 'direita':128,'spin':386, 'esquerda':64}
possible_states = [format_state(s) for s in itertools.product([0, 1], repeat = 10)]
mostrar = True

class MarioAI:
    def __init__(self, sensors = 10):
        self.sensors = sensors
        self.policy = self._generate_policy()
        self.fitness = None

    def get_policy(self):
        return self.policy

    def get_fitness(self):
        if self.fitness == None:
            self.fitness = self._calc_fitness()
        
        return self.fitness

    def update_policy(self, state, action):
        self.policy[state] = action

    def _generate_policy(self):
        policy = dict()
        
        for state in list(itertools.product([0, 1], repeat = self.sensors)):
            random_action = random.choice(list(moves))
            policy[format_state(state)] = random_action

        return policy

    def _calc_fitness(self):
        env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1', players=1)    
        env.reset()
        
        fitness = 0
        while not env.data.is_done(): 
            state, x_pos, _ = self.get_current_state(env)
            action = self.evaluate(state)
            
            performAction(moves[action], env)

            fitness = max(fitness, x_pos)
        
            print(fitness)
            if mostrar:
                env.render()
        

        return fitness

    def evaluate(self, state):
        print('state', state)
        print('policy', self.policy[state])
        return self.policy[state]

    def get_current_state(self, env):
        state, x_pos, y_pos = getInputs(getRam(env))
        state = state.reshape(13, 13)
    
        transformed_states = {
            'upper_left':   self.has_enemies(state, 1, 4, 0, 4),
            'mid_left':     self.has_enemies(state, 1, 4, 5, 7),
            'upper_left':   self.has_enemies(state, 1, 4, 0, 4),
            'lower_left':   self.has_enemies(state, 1, 4, 8, 12),

            'up':           self.has_enemies(state, 5, 7, 0, 4),
            'down':         self.has_enemies(state, 5, 7, 8, 12),

            'upper_right1': self.has_enemies(state, 8, 12, 0, 2), 
            'upper_right2': self.has_enemies(state, 8, 12, 3, 4),
            'near_right':   self.has_enemies(state, 8, 10, 5, 7),
            'far_right':    self.has_enemies(state, 11, 12, 5, 7), 
            'lower_right':  self.has_enemies(state, 8, 12, 8, 12),            
        }

        print(format_state(transformed_states.values()))
        return format_state(transformed_states.values()), x_pos, y_pos

    def has_enemies(self, state, col_beg, col_end, row_beg, row_end):
        found_enemy = False
        enemy = -1
        for row in range(row_beg, row_end + 1):
            for col in range(col_beg, col_end + 1):
                found_enemy |= (state[row][col] == enemy)
        
        return found_enemy

class GeneticAlgorithm:
    def __init__(self, generations = 100, population_size = 2):
        self.population_size = population_size
        self.generations = generations
    
        self.population = self.generatePopulation()
        #print(self.population)

    def generatePopulation(self):
        return [MarioAI() for _ in range(self.population_size)]

    def crossover_individuals(self, ind_a, ind_b):
        crossover_rate = random.randint(0, len(ind_a))
        
        policy_a = ind_a.policy
        policy_b = ind_b.policy
        target_states = random.sample(possible_states, crossover_rate)

        for ts in target_states:
            ind_a.update_policy(ts, policy_b[ts])
            ind_b.update_policy(ts, policy_a[ts])

    def mutate_individual(self, individual):
        target_state = random.choice(possible_states)
        mutated_action = random.choice(list(moves))
        print(target_state, mutated_action)
        individual.update_policy(target_state, mutated_action)
    
    def tournament(self, population):
        pass

    def select_best(self, population, selection_rate):
        pass

    def train(self):
        pass

def main():  
    global mostrar
    global env
  
    mostrar = True
    
if __name__ == "__main__":
    main()