# Blackjack Neuroevolution Agent (BIAI)

## Overview

This project trains a neural network to play Blackjack using a Genetic Algorithm (GA) — a technique known as **neuroevolution**. Instead of backpropagation, the agent's weights are evolved over generations by selecting, crossing over, and mutating the best-performing individuals in a population.

The environment used is `Blackjack-v1` from [Gymnasium](https://gymnasium.farama.org/).

---

## Project Structure

```
Blackjack_BIAI/
├── DATA/                    # Generated plots and datasets
│   └── *.png                # Fitness evolution and analysis plots
├── DOC/                     # Documentation and presentations
├── IMP/                     # Source code and implementation
│   ├── main.py              # Entry point — orchestrates training and playback
│   ├── analyze.py           # Analysis pipeline for operator comparison
│   ├── genetic_algorithm.py # GA logic: population, fitness, selection, crossover, mutation
│   ├── neural_network.py    # Feedforward neural network (agent controller)
│   ├── visualization.py     # Live training display, game sidebar, fitness graphs
│   ├── test_agent.py        # Utility to test a previously saved agent
│   └── best_agent.npy       # Saved agent weights (generated after training)
└── requirements.txt         # Python dependencies
```

---

## How It Works

### Neural Network Architecture

A simple feedforward network maps the game state to an action (Hit or Stand):

```
Input (3)  →  Hidden (16, ReLU)  →  Output (1, Sigmoid)
```

| Input feature | Description |
|---|---|
| `player_sum` | Current total of the player's hand |
| `dealer_card` | Dealer's visible card |
| `usable_ace` | Whether the player has a usable ace |

The output is thresholded at 0.5: values above → **Hit (1)**, below → **Stand (0)**.

The full network has **81 weights**, which form each individual's DNA in the GA.

### Genetic Algorithm

Each generation follows this cycle:

1. **Evaluate** — each individual plays 100 games; average reward is the fitness score.
2. **Select** — choose parents using tournament or rank-based selection.
3. **Crossover** — uniform crossover combines genes from two parents.
4. **Mutate** — Gaussian noise is added to random genes with probability `MUTATION_RATE`.
5. **Elitism** — the best individual from the current generation is always preserved.

#### Selection Methods

| Method | Description |
|---|---|
| `tournament` | Picks the best of K randomly sampled individuals (default K=3). Fast and effective. |
| `rank` | Assigns selection probability proportional to rank. Robust to negative fitness values. |

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd Blackjack_BIAI
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

**Important:** All code should now be executed from inside the `IMP` folder!

```bash
cd IMP
```

### Train a new agent

```bash
python main.py
```

Before running, configure the mode at the top of `main.py`:

```python
SELECTION_METHOD = "tournament"  # "tournament" or "rank"
COMPARE_MODE = False             # True: runs both methods and plots side-by-side
```

**Single method** (`COMPARE_MODE = False`):
1. Opens a live pygame window showing generation progress, fitness curves, and elapsed time.
2. Runs the GA for 100 generations.
3. Displays a final summary screen before playback.
4. Saves the best agent weights to `best_agent.npy` (inside `IMP`).
5. Saves a fitness evolution graph as `fitness_evolution_{method}.png` (inside `DATA`).
6. Plays 5 visual games with the best trained agent.

**Compare mode** (`COMPARE_MODE = True`):
1. Runs tournament selection followed by rank selection in sequence.
2. Saves a side-by-side comparison graph as `fitness_evolution_comparison.png` (inside `DATA`).
3. Plays back the best agent from whichever method scored higher.

### Run the Analysis Pipeline

```bash
python analyze.py
```

Runs the full experimental suite (mutation rates, crossover variants, and grid search) generating all the analysis plots inside the `DATA` folder.

### Test a pre-trained agent

```bash
python test_agent.py
```

Loads `best_agent.npy` from disk and plays 5 visual games with it, without re-running training.

---

## Configuration

Key hyperparameters are defined at the top of `genetic_algorithm.py`:

| Parameter | Default | Description |
|---|---|---|
| `POPULATION_SIZE` | 50 | Number of individuals per generation |
| `GENERATIONS` | 100 | Number of training generations |
| `MUTATION_RATE` | 0.1 | Probability of mutating each gene |
| `N_GAMES` | 100 | Games played per fitness evaluation |
| `TOURNAMENT_SIZE` | 3 | Candidates sampled in tournament selection |

---

## Output Files

| File | Location | Description |
|---|---|---|
| `best_agent.npy` | `IMP/` | Saved weights of the best evolved agent |
| `fitness_evolution_*.png` | `DATA/` | Fitness graphs from basic runs |
| `mutation_analysis.png` | `DATA/` | Analysis plot from `analyze.py` phase 1 |
| `crossover_analysis_*.png` | `DATA/` | Analysis plots from `analyze.py` phase 2 |
| `best_combination.png` | `DATA/` | Grid search results from `analyze.py` phase 3 |

> Note: The output graphs and generated models are excluded from version control via `.gitignore`.
