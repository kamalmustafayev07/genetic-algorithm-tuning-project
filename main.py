from src.optimization.ga_evaluator import evaluate_solution
from src.optimization.ga import genetic_algorithm
from src.utils.config import GA_PARAMS
from src.utils.saver import save_json, plot_history
import os

BEST_PATH = "results/metrics/best_params.json"
PLOT_PATH = "results/figures/ga_progress.png"

if __name__ == "__main__":
    best_vector, best_fitness, history = genetic_algorithm(
        objective_function=evaluate_solution,
        pop_size=GA_PARAMS["pop_size"],
        elite_size=GA_PARAMS["elite_size"],
        generations=GA_PARAMS["generations"],
        crossover_rate=GA_PARAMS["crossover_rate"],
        mutation_rate=GA_PARAMS["mutation_rate"],
        mutation_strength=GA_PARAMS["mutation_strength"],
        patience=GA_PARAMS["patience"],
    )
    save_json(BEST_PATH, {"best_vector": best_vector, "best_fitness": best_fitness})
    plot_history(history, "GA Optimization Progress", PLOT_PATH)