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

moves = {'runright':130, 'runjumpright':131, 'right':128,'runspinright':386, 'left':64}
possible_states = [format_state(s) for s in itertools.product([0, 1], repeat = 10)]
mostrar = True

class MarioAI:
    def __init__(self, env, sensors = 10, policy = None):
        self.env = env
        self.sensors = sensors
        self.policy = policy
        self.used_policy = set()

        if self.policy == None:
            self.policy = self.__generate_policy()
        
        self.fitness = self.__calc_fitness()

    def get_policy(self):
        return self.policy.copy()

    def get_used_policy(self):
        return self.used_policy.copy()

    def get_fitness(self):
        if self.fitness == None:
            self.fitness = self.__calc_fitness()
        
        return self.fitness

    def __generate_policy(self):
        policy = dict()
        
        for state in list(itertools.product([0, 1], repeat = self.sensors)):
            random_action = random.choice(list(moves))
            policy[format_state(state)] = random_action

        return policy

    def __calc_fitness(self):
        self.env.reset()
        self.used_policy.clear()

        fitness = 0

        stuck_count, timeout_count = 0, 0
        last_xpos, last_ypos = 0, 0
        while timeout_count < 100 and not self.env.data.is_done(): 
            state, xpos, ypos = self.__get_current_state()
            timeout_count = timeout_count + 1 if xpos <= fitness else 0
            stuck_count = stuck_count + 1 if xpos == last_xpos else 0

            self.__perfom_action(state, ypos, last_ypos, stuck_count)

            fitness = max(fitness, xpos)
            last_ypos, last_xpos = ypos, xpos

            if mostrar:
                self.env.render()

        print("Fitness: ", fitness)
        return fitness

    def __perfom_action(self, state, ypos, last_ypos, stuck_count):
        self.used_policy.add(state)
    
        if stuck_count > 25:
            performAction(0, self.env)
            for _ in range(6):
                performAction(moves['runjumpright'], self.env)
        else:
            action = self.__evaluate(state)
            if action == 'runjumpright' and ypos == last_ypos:
                performAction(0, self.env)
            performAction(moves[action], self.env)

    def __evaluate(self, state):
        return self.policy[state]

    def __get_current_state(self):
        state, x_pos, y_pos = getInputs(getRam(self.env))
        state = state.reshape(13, 13)
    
        transformed_states = {
            'upper_left':   self.__has_enemies(state, 1, 4, 0, 4),
            'mid_left':     self.__has_enemies(state, 1, 4, 5, 7),
            'upper_left':   self.__has_enemies(state, 1, 4, 0, 4),
            'lower_left':   self.__has_enemies(state, 1, 4, 8, 12),

            'up':           self.__has_enemies(state, 5, 7, 0, 4),
            'down':         self.__has_enemies(state, 5, 7, 8, 12),

            'upper_right1': self.__has_enemies(state, 8, 12, 0, 2), 
            'upper_right2': self.__has_enemies(state, 8, 12, 3, 4),
            'near_right':   self.__has_enemies(state, 8, 10, 5, 7),
            'far_right':    self.__has_enemies(state, 11, 12, 5, 7), 
            'lower_right':  self.__has_enemies(state, 8, 12, 8, 12),            
        }

        return format_state(transformed_states.values()), x_pos, y_pos

    def __has_enemies(self, state, col_beg, col_end, row_beg, row_end):
        found_enemy = False
        enemy = -1
        for row in range(row_beg, row_end + 1):
            for col in range(col_beg, col_end + 1):
                found_enemy |= (state[row][col] == enemy)
        
        return found_enemy

class GeneticAlgorithm:
    def __init__(self, generations = 25, population_size = 10):
        self.env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1', players=1)    
        self.population_size = population_size
        self.generations = generations
    
        self.population = self.generatePopulation()
        self.best_fitness = 0
        #print(self.population)

    def generatePopulation(self):
        return [MarioAI(env = self.env) for _ in range(self.population_size)]

    def __crossover(self, population):
        print("Starting crossover...")
        removal_rate = int(0.2 * self.population_size)
        target_p = self.__remove_worst(population, removal_rate)

        crossed_p = []
        while len(crossed_p) < self.population_size:
            [ind_a, ind_b] = random.sample(target_p, 2)
            crossed_a, crossed_b = self.__recombine_individuals(ind_a, ind_b)
            crossed_p += [crossed_a, crossed_b]
        
        return crossed_p

    def __recombine_individuals(self, ind_a, ind_b):
        possible_states = list(ind_a.get_used_policy()) + list(ind_b.get_used_policy())
        crossover_rate = random.randint(0, len(possible_states))
        target_states = random.sample(possible_states, crossover_rate)

        policy_a = ind_a.get_policy()
        policy_b = ind_b.get_policy()

        for ts in target_states:
            policy_a[ts], policy_b[ts] = policy_b[ts], policy_a[ts]

        return MarioAI(env = self.env, policy = policy_a), MarioAI(env = self.env, policy = policy_b)

    def __mutation(self, population):
        print("Starting mutation...")
        return [self.__mutate_individual(ind) for ind in population]

    def __mutate_individual(self, individual):
        target_state = random.choice(list(individual.get_used_policy()))
        mutated_action = random.choice(list(moves))
        
        policy = individual.get_policy()
        policy[target_state] = mutated_action

        return MarioAI(env = self.env, policy = policy)
    
    def __selection(self, population):
        print("Starting selection...")
        selection_rate = int(0.2 * self.population_size)
        tournament_rate = int(0.5 * self.population_size)
        
        selected = []
        while len(selected) < self.population_size:
            p_sample = random.sample(population, tournament_rate)
            best_individuals = self.__select_best(p_sample, selection_rate)
            selected += best_individuals
            
        return selected

    def __select_best(self, population, selection_rate):
        population.sort(key = lambda ind : ind.get_fitness(), reverse = True)
        return population[0:selection_rate]

    def __remove_worst(self, population, selection_rate):
        population.sort(key = lambda ind : ind.get_fitness())
        return population[selection_rate:]

    def train(self):
        for gen in range(self.generations):
            mutated = self.__mutation(self.population)
            crossed = self.__crossover(mutated)
            selected = self.__selection(self.population + crossed)

            self.population = selected + self.__select_best(crossed, 1)
            self.best_fitness = self.__select_best(self.population, 1)[0].get_fitness()

            print("Generation {0}. Best Fitness = {1}".format(gen + 1, self.best_fitness))
            print("Fitness of the current generation: ", [p.get_fitness() for p in self.population])

def main():  
    global mostrar
    global env
  
    mostrar = True

    ga = GeneticAlgorithm()
    ga.train()
    
if __name__ == "__main__":
    main()