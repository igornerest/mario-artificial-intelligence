

import sys
import os
import pickle
import retro
import time
from rominfo import *
from utils import *
import random
import itertools
import copy

def format_state(state):
    return ','.join(map(lambda x : str(int(x)), state))  

moves = {'corre':130, 'pula':131, 'direita':128,'spin':386, 'esquerda':64}
possible_states = [format_state(s) for s in itertools.product([0, 1], repeat = 10)]
mostrar = True

class MarioAI:
    def __init__(self, env, sensors = 10, policy = None):
        self.env = env
        self.sensors = sensors
        self.policy = policy
        self.used_policy = set()

        if self.policy == None:
            self.policy = self._generate_policy()
        
        self.fitness = self._calc_fitness()

    def get_policy(self):
        return self.policy.copy()

    def get_used_policy(self):
        return self.used_policy.copy()

    def get_fitness(self):
        if self.fitness == None:
            self.fitness = self._calc_fitness()
        
        return self.fitness

    def _generate_policy(self):
        policy = dict()
        
        for state in list(itertools.product([0, 1], repeat = self.sensors)):
            random_action = random.choice(list(moves))
            policy[format_state(state)] = random_action

        return policy

    def _calc_fitness(self):
        self.env.reset()
        self.used_policy.clear()

        max_it = 0
        max_x_pos = 1
        fitness = 0
        while max_it < 100 and not self.env.data.is_done(): 
            state, x_pos, _ = self._get_current_state()
            action = self.evaluate(state)
            self.used_policy.add(state)

            performAction(moves[action], self.env)
            fitness = max(fitness, x_pos)
            max_it = max_it + 1 if x_pos <= max_x_pos else 0
            max_x_pos = max(max_x_pos, x_pos)

            if mostrar:
                self.env.render()

        print("calculated fitness: ", fitness)
        return fitness

    def evaluate(self, state):
        #print('state', state)
        #print('policy', self.policy[state])
        return self.policy[state]

    def _get_current_state(self):
        state, x_pos, y_pos = getInputs(getRam(self.env))
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

        #print(format_state(transformed_states.values()))
        return format_state(transformed_states.values()), x_pos, y_pos

    def has_enemies(self, state, col_beg, col_end, row_beg, row_end):
        found_enemy = False
        enemy = -1
        for row in range(row_beg, row_end + 1):
            for col in range(col_beg, col_end + 1):
                found_enemy |= (state[row][col] == enemy)
        
        return found_enemy

class GeneticAlgorithm:
    def __init__(self, generations = 20, population_size = 10):
        self.env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1', players=1)    
        self.population_size = population_size
        self.generations = generations
    
        self.population = self.generatePopulation()
        self.best_fitness = 0
        #print(self.population)

    def generatePopulation(self):
        return [MarioAI(env = self.env) for _ in range(self.population_size)]

    def _crossover(self, population):
        print("Crossover")
        removal_rate = int(0.2 * self.population_size)
        target_p = self._remove_worst(population, removal_rate)

        crossed_p = []
        while len(crossed_p) < self.population_size:
            [ind_a, ind_b] = random.sample(target_p, 2)
            crossed_a, crossed_b = self._recombine_individuals(ind_a, ind_b)
            crossed_p += [crossed_a, crossed_b]
        
        return crossed_p

    def _recombine_individuals(self, ind_a, ind_b):
        possible_states = list(ind_a.get_used_policy()) + list(ind_b.get_used_policy())
        crossover_rate = random.randint(0, len(possible_states))
        target_states = random.sample(possible_states, crossover_rate)

        policy_a = ind_a.get_policy()
        policy_b = ind_b.get_policy()

        for ts in target_states:
            policy_a[ts], policy_b[ts] = policy_b[ts], policy_a[ts]

        return MarioAI(env = self.env, policy = policy_a), MarioAI(env = self.env, policy = policy_b)

    def _mutation(self, population):
        print("mutation")
        return [self._mutate_individual(ind) for ind in population]

    def _mutate_individual(self, individual):
        target_state = random.choice(list(individual.get_used_policy()))
        mutated_action = random.choice(list(moves))
        
        policy = individual.get_policy()
        policy[target_state] = mutated_action

        return MarioAI(env = self.env, policy = policy)
    
    def _selection(self, population):
        print("selection")
        selection_rate = int(0.2 * self.population_size)
        tournament_rate = int(0.5 * self.population_size)
        
        selected = []
        while len(selected) < self.population_size:
            p_sample = random.sample(population, tournament_rate)
            best_individuals = self._select_best(p_sample, selection_rate)
            selected += best_individuals
            
        return selected

    def _select_best(self, population, selection_rate):
        population.sort(key = lambda ind : ind.get_fitness(), reverse = True)
        #print([p.get_fitness() for p in population])
        return population[0:selection_rate]

    def _remove_worst(self, population, selection_rate):
        population.sort(key = lambda ind : ind.get_fitness())
        #print([p.get_fitness() for p in population])
        return population[selection_rate:]

    def train(self):
        for gen in range(self.generations):
            mutated = self._mutation(self.population)
            crossed = self._crossover(mutated)
            selected = self._selection(self.population + crossed)

            self.population = selected
            self.best_fitness = self._select_best(self.population, 1)[0].get_fitness()

            print("Finished generation {0}. Best Fitness = {1}".format(gen, self.best_fitness))
            print("Fitness of the current generation: ", [p.get_fitness() for p in self.population])
        #value = set(new_population[0].get_policy().items()) ^set(self.population[0].get_policy().items()) 

        #print(value)

def main():  
    global mostrar
    global env
  
    mostrar = True

    ga = GeneticAlgorithm()
    ga.train()
    
if __name__ == "__main__":
    main()