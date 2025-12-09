# CNN Hyperparameter Optimization using Genetic Algorithm

A Python project that uses Genetic Algorithms (GA) to automatically optimize CNN architecture hyperparameters for the MNIST digit classification task.

## Overview

This project implements a genetic algorithm to search for optimal hyperparameters of a Convolutional Neural Network (CNN). Instead of manual tuning, the GA explores the hyperparameter space to find configurations that maximize validation accuracy on the MNIST dataset.

## Features

- **Automated Hyperparameter Search**: Uses GA to optimize 8 key CNN hyperparameters
- **Early Stopping**: Implements patience mechanism to prevent unnecessary computations
- **Comprehensive Logging**: Tracks all evaluations and parameter configurations
- **Visualization**: Generates fitness history plots
- **Modular Architecture**: Clean separation of concerns with well-organized modules

## Project Structure

```
project/
├── configs/
│   ├── training.yaml      # Training configuration
│   ├── ga.yaml            # Genetic algorithm parameters
│   └── bounds.yaml        # Hyperparameter search bounds
├── src/
│   ├── data/
│   │   └── data_loader.py     # MNIST data loading and preprocessing
│   ├── models/
│   │   ├── model_builder.py   # CNN architecture builder
│   │   └── cnn_model.py       # Model evaluation function
│   ├── optimization/
│   │   └── ga.py              # Genetic algorithm implementation
│   └── utils/
│       ├── config.py          # Configuration loader
│       ├── logging.py         # Logging utilities
│       └── saver.py           # Result saving and plotting
└── README.md
```

## Optimized Hyperparameters

The GA optimizes the following CNN hyperparameters:

1. **filters1**: Number of filters in first Conv2D layer
2. **kernel1**: Kernel size for first Conv2D layer
3. **filters2**: Number of filters in second Conv2D layer
4. **kernel2**: Kernel size for second Conv2D layer
5. **dropout_rate**: Dropout rate between Conv and Dense layers
6. **dense1**: Number of neurons in first Dense layer
7. **dense2**: Number of neurons in second Dense layer
8. **learning_rate**: Adam optimizer learning rate (log scale)

## CNN Architecture

The model architecture consists of:
- Two Conv2D layers with ReLU activation and MaxPooling
- Dropout layer for regularization
- Two Dense hidden layers with ReLU activation
- Output Dense layer with softmax activation (10 classes)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install tensorflow scikit-learn pyyaml matplotlib numpy
```

## Configuration

Configure the optimization process using YAML files in the `configs/` directory:

### bounds.yaml
```yaml
bounds:
  filters1:       [10, 30]
  kernel1:        [2, 3]
  filters2:       [10, 20]
  kernel2:        [2, 3]
  dropout_rate:   [0.1, 0.3]
  dense1:         [32, 64]
  dense2:         [16, 32]
  learning_rate:  [-5, -3]
```

### ga.yaml
```yaml
pop_size: 20
elite_size: 2
generations: 14
patience: 5
crossover_rate: 0.9
mutation_rate: 0.1
mutation_strength: 0.1
patience: 5
```

### training.yaml
```yaml
epochs_ga: 10
batch_size: 128
```

## Usage

```python
from src.optimization.ga import genetic_algorithm
from src.models.cnn_model import evaluate_solution

# Run genetic algorithm
best_individual, best_fitness, history = genetic_algorithm(
    objective_function=evaluate_solution,
    pop_size=20,
    elite_size=2,
    generations=14,
    patience=5
)

# Best fitness represents (1 - validation_accuracy)
print(f"Best validation accuracy: {1 - best_fitness:.4f}")
```

## How It Works

1. **Initialization**: Creates a random population of hyperparameter configurations
2. **Evaluation**: Each configuration is used to build and train a CNN model
3. **Selection**: Tournament selection chooses parents based on fitness
4. **Crossover**: Parent configurations are combined to create offspring
5. **Mutation**: Random perturbations introduce variation
6. **Elitism**: Best configurations are preserved across generations
7. **Early Stopping**: Process halts if no improvement for `patience` generations

## Genetic Algorithm Details

- **Fitness Function**: `1 - validation_accuracy` (minimize)
- **Selection**: Tournament selection (size=3)
- **Crossover**: Uniform crossover with 90% rate
- **Mutation**: Gaussian mutation with adaptive strength
- **Elitism**: Top 2 individuals preserved

## Output

The algorithm provides:
- Real-time progress tracking with generation-by-generation updates
- Best hyperparameter configuration found
- Best validation accuracy achieved
- Fitness history for visualization

Example output:
```
Generation 1, Best fitness: 0.04523
Evaluation 1:
Params: filters1=32, kernel1=3, filters2=64, kernel2=3, dropout=0.250, dense1=128, dense2=64, lr=0.001000
Validation accuracy: 0.9548 (loss: 0.0452)
...
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- scikit-learn
- PyYAML
- Matplotlib

## Performance Considerations

- Each generation requires training multiple CNN models
- GPU acceleration highly recommended
- Memory cleanup implemented to prevent memory leaks
- Early stopping reduces unnecessary computations

## Future Improvements

- [ ] Add support for different CNN architectures
- [ ] Implement parallel evaluation of individuals
- [ ] Add support for other optimization algorithms (PSO, Bayesian Optimization)
- [ ] Include more advanced mutation strategies
- [ ] Add cross-validation for more robust evaluation

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

```bibtex
@software{cnn_ga_optimization,
  title = {CNN Hyperparameter Optimization using Genetic Algorithm},
  author = {[Kamal Mustafayev]},
  year = {2025},
  note = {University Research Project}
}
```
