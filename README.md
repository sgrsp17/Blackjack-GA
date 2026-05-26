# Blackjack Neuroevolution Agent (BIAI)

## Overview
This project implements an Artificial Intelligence agent designed to play the Blackjack environment provided by Gymnasium (`Blackjack-v1`). The core objective is to develop a robust neural network controller trained via neuroevolution, specifically utilizing a Genetic Algorithm for global optimization.

## Project Structure
- `main.py`: Entry point that orchestrates the training pipeline.
- `genetic_algorithm.py`: Implements the GA logic — population creation, fitness evaluation, selection, crossover, mutation, and the training loop.
- `neural_network.py`: Simple feedforward neural network that serves as the agent's decision-making controller.
- `visualization.py`: Handles visualization during best-agent playback — draws game results sidebar and fitness evolution graph.

## Setup and Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd Blackjack-GA
```

### 2. Create a virtual environment (recommended)
```bash
python -3.11 -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Train a new agent
```bash
python main.py
```

This will:
1. Train the GA for 100 generations (configurable in `genetic_algorithm.py`)
2. Save the best agent weights to `best_agent.npy`
3. Display a fitness evolution graph and save it as `fitness_evolution.png`
4. Play 5 games with the best trained agent with visual rendering

### Use a pre-trained agent
Modify the code to load `best_agent.npy` and call `evaluate_fitness()` with the loaded weights.

## Configuration
Key hyperparameters are defined at the top of `genetic_algorithm.py`:
- `POPULATION_SIZE`: Number of individuals per generation (default: 50)
- `GENERATIONS`: Number of training generations (default: 100)
- `MUTATION_RATE`: Probability of gene mutation (default: 0.1)
- `N_GAMES`: Games per fitness evaluation (default: 100)
- `TOURNAMENT_SIZE`: Individuals in tournament selection (default: 3)
