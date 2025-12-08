import numpy as np
import random
from src.utils.config import BOUNDS as bounds, GA_PARAMS 

def initialize_population(pop_size, bounds):
    population = []
    for _ in range(pop_size):
        individual = [random.uniform(lower, upper) for lower, upper in bounds]
        population.append(individual)
    return population


def evaluate_population(population, objective_function):
    evaluated = []
    for ind in population:
        fitness = objective_function(ind)
        evaluated.append((ind, fitness))
    evaluated.sort(key=lambda x: x[1])
    return evaluated


def select_parents(evaluated_population, tournament_size=3):
    def tournament():
        candidates = random.sample(evaluated_population, tournament_size)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    parent1 = tournament()
    parent2 = tournament()
    return parent1, parent2


def crossover(parent1, parent2, crossover_rate=0.9):
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]
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
            mutated[i] = max(lower, min(upper, mutated[i]))
    return mutated


def genetic_algorithm(
    objective_function,
    pop_size=20,
    elite_size=2,
    generations=14,
    patience=5,  
    crossover_rate=0.9,
    mutation_rate=0.1,
    mutation_strength=0.1):

    population = initialize_population(pop_size, bounds)
    evaluated_pop = evaluate_population(population, objective_function)
    best_individual, best_fitness = evaluated_pop[0]

    best_fitness_history = [best_fitness]
    no_improve_count = 0 

    for gen in range(generations):
        print(f"Generation {gen+1}, Best fitness: {best_fitness:.5f}")

        elites = [ind for ind, fit in evaluated_pop[:elite_size]]

        new_population = elites[:]
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(evaluated_pop)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate, bounds, mutation_strength)
            child2 = mutate(child2, mutation_rate, bounds, mutation_strength)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        evaluated_pop = evaluate_population(new_population, objective_function)
        current_best, current_fitness = evaluated_pop[0]

        if current_fitness < best_fitness:
            best_individual, best_fitness = current_best, current_fitness
            best_fitness_history.append(best_fitness)
            no_improve_count = 0
            print(f"✔ Improved fitness: {best_fitness:.5f}")
        else:
            no_improve_count += 1
            best_fitness_history.append(best_fitness)
            print(f"✖ No improvement ({no_improve_count}/{patience})")

        if no_improve_count >= patience:
            print("\n--- EARLY STOPPING: no improvement ---")
            break

    print("\n=== Genetic Algorithm Finished ===")
    print(f"Best Individual (parameters): {best_individual}")
    print(f"Best Fitness (loss): {best_fitness}")
    print(f"Best Accuracy: {1 - best_fitness:.4f}")

    return best_individual, best_fitness, best_fitness_history

