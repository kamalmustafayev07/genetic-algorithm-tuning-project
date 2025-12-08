import json
import os
import matplotlib.pyplot as plt
from src.ga import genetic_algorithm
from src.cnn_model import evaluate_solution

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

def run_optimization():
    # Run the GA with the objective function
    best_params, best_loss, history = genetic_algorithm(
        objective_function=evaluate_solution,
        pop_size=20,
        elite_size=2,
        generations=1,
        crossover_rate=0.9,
        mutation_rate=0.1,
        mutation_strength=0.1
    )

    # Calculate accuracy from loss
    best_accuracy = 1 - best_loss

    # Print results
    print(f"Best parameters: {best_params}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

    # Save best parameters to JSON
    with open('results/best_params.json', 'w') as f:
        json.dump({
            'params': best_params,
            'loss': best_loss,
            'accuracy': best_accuracy
        }, f, indent=4)

    # Plot fitness history
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(history)), history, marker='o')
    plt.xlabel('Generation (0 = Initial)')
    plt.ylabel('Best Loss (1 - Val Accuracy)')
    plt.title('GA Optimization Progress')
    plt.grid(True)
    plt.savefig('results/ga_progress.png')
    plt.close()  # Close the figure to free memory

    # Optional: Save history to JSON for further analysis
    with open('results/fitness_history.json', 'w') as f:
        json.dump(history, f)

if __name__ == "__main__":
    # Redirect output to a log file (optional, for long runs)
    # import sys
    # sys.stdout = open('results/run_log.txt', 'w')
    
    run_optimization()