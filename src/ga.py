import numpy as np
import random

# Define the bounds as per the project
bounds = [
    (10, 30),    # x0: number of filters in convolutional layer 1
    (2, 3),      # x1: kernel size in convolutional layer 1
    (10, 20),    # x2: number of filters in convolutional layer 2
    (2, 3),      # x3: kernel size in convolutional layer 2
    (0.1, 0.3),  # x4: dropout_rate
    (32, 64),    # x5: number of neurons in dense layer 1
    (16, 32),    # x6: number of neurons in dense layer 2
    (-5, -3)     # x7: log10_learning_rate
]

def initialize_population(pop_size, bounds):
    # Empty array
    population = []
    for _ in range(pop_size):
        # generating random values for individuals within the given bounds
        individual = [random.uniform(lower, upper) for lower, upper in bounds]
        # appending each individual to the population
        population.append(individual)
    return population

def evaluate_population(population, objective_function):
    evaluated = []
    for ind in population:
        fitness = objective_function(ind)
        evaluated.append((ind, fitness))
    evaluated.sort(key=lambda x: x[1])  # Sort by fitness ascending (minimize)
    return evaluated

def select_parents(evaluated_population, tournament_size=3):
    def tournament():
        candidates = random.sample(evaluated_population, tournament_size)
        candidates.sort(key=lambda x: x[1])  # Sort by fitness
        return candidates[0][0]  # Best in tournament
    parent1 = tournament()
    parent2 = tournament()
    return parent1, parent2

def crossover(parent1, parent2, crossover_rate=0.9):
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]  # No crossover
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

def mutate(individual, mutation_rate=0.1, bounds=bounds, mutation_strength=0.1):
    mutated = individual[:]
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            lower, upper = bounds[i]
            sigma = (upper - lower) * mutation_strength
            mutated[i] += random.gauss(0, sigma)
            mutated[i] = max(lower, min(upper, mutated[i]))  # Clip to bounds
    return mutated

def genetic_algorithm(objective_function, pop_size=20, elite_size=2, generations=15, crossover_rate=0.9, mutation_rate=0.1, mutation_strength=0.1):
    # Initialize population
    population = initialize_population(pop_size, bounds)
    evaluated_pop = evaluate_population(population, objective_function)
    best_individual, best_fitness = evaluated_pop[0]
    
    # Track best fitness history (starting with initial)
    best_fitness_history = [best_fitness]
    
    for gen in range(generations):
        print(f"Generation {gen+1}")
        # Elites
        elites = [ind for ind, fit in evaluated_pop[:elite_size]]
        # Generate new population
        new_population = elites[:]
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(evaluated_pop)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate, bounds, mutation_strength)
            child2 = mutate(child2, mutation_rate, bounds, mutation_strength)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        # Evaluate new population
        evaluated_pop = evaluate_population(new_population, objective_function)
        current_best, current_fitness = evaluated_pop[0]
        if current_fitness < best_fitness:
            best_individual, best_fitness = current_best, current_fitness
            print(f"New best fitness: {best_fitness}")
        
        # Append the best fitness so far to history
        best_fitness_history.append(best_fitness)
    
    return best_individual, best_fitness, best_fitness_history