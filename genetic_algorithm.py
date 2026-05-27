import numpy as np
import random
from neural_network import NeuralNetwork
from visualization import draw_game_summary

# --- Genetic Algorithm Constants ---
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1
DNA_SIZE = 81  # 3*16 + 16 + 16*1 + 1 (For 3 inputs, 16 hidden, 1 output)
N_GAMES = 100
TOURNAMENT_SIZE = 3


def create_population(size, dna_size):
    """Creates the initial population with random weights between -1 and 1."""
    return [np.random.uniform(-1, 1, dna_size) for _ in range(size)]


def evaluate_fitness(individual, env, n_games=N_GAMES, verbose=False, agent_type="thinking"):
    """Evaluates an individual by playing N_GAMES and returning the average reward."""
    nn = NeuralNetwork(weights=individual)
    total_reward = 0.0

    wins = 0
    losses = 0
    draws = 0
    game_results = []

    for i in range(n_games):
        observation, info = env.reset()
        finished = False
        truncated = False

        while not (finished or truncated):
            # The observation in blackjack is (player_sum, dealer_card, usable_ace)
            if agent_type == "random":
                action = env.action_space.sample()
            else:
                action = nn.predict(observation)

            if action not in [0, 1]:
                action = 1 if action > 0.5 else 0

            observation, reward, finished, truncated, info = env.step(action)

            if finished or truncated:
                total_reward += reward
                if reward > 0:
                    wins += 1
                    resultado_texto = "WON!"
                    cor = (0, 255, 0)
                    result_char = 'W'
                elif reward < 0:
                    losses += 1
                    resultado_texto = "LOST!"
                    cor = (255, 0, 0)
                    result_char = 'L'
                else:
                    draws += 1
                    resultado_texto = "DRAW"
                    cor = (255, 255, 0)
                    result_char = 'D'
                game_results.append(result_char)

                if getattr(env, "render_mode", None) == "human":
                    import pygame
                    pygame.font.init()
                    fonte = pygame.font.SysFont('Arial', 64, bold=True)
                    texto_surface = fonte.render(resultado_texto, True, cor)

                    screen = pygame.display.get_surface()
                    if screen is not None:
                        rect = texto_surface.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
                        bg_rect = pygame.Surface((rect.width + 20, rect.height + 20))
                        bg_rect.set_alpha(150)
                        bg_rect.fill((0, 0, 0))
                        screen.blit(bg_rect, (rect.x - 10, rect.y - 10))
                        screen.blit(texto_surface, rect)
                        draw_game_summary(screen, game_results)
                        pygame.display.update()
                    import time
                    time.sleep(1.0)

        if verbose:
            print(f"Game {i+1}: {game_results[-1] if game_results else 'N/A'}")

    if verbose:
        print(f"\n--- Final Performance ({n_games} Games) ---")
        print(f"Wins:   {wins}")
        print(f"Losses: {losses}")
        print(f"Draws:  {draws}")
        print("-----------------------------------")

    return total_reward / n_games


def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    """Selects the best individual among K randomly chosen individuals."""
    selected_indices = random.sample(range(len(population)), k)
    best_index = selected_indices[0]
    best_fitness = fitnesses[best_index]

    for idx in selected_indices[1:]:
        if fitnesses[idx] > best_fitness:
            best_fitness = fitnesses[idx]
            best_index = idx

    return population[best_index]


def rank_selection(population, fitnesses):
    """Selects an individual using linear rank-based probability.

    Ranks individuals from worst (rank 1) to best (rank N) and assigns
    selection probability proportional to rank, avoiding issues with
    negative fitness values that break roulette-wheel selection.
    """
    n = len(population)
    sorted_indices = sorted(range(n), key=lambda i: fitnesses[i])
    # rank[i] = position in sorted order (1-based), so best gets rank N
    ranks = [0] * n
    for rank, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank
    total = n * (n + 1) / 2
    probabilities = [ranks[i] / total for i in range(n)]
    chosen = random.choices(range(n), weights=probabilities, k=1)[0]
    return population[chosen]


def crossover(parent1, parent2):
    """Uniform crossover: randomly chooses genes from both parents."""
    child = np.zeros_like(parent1)
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child


def mutate(individual, mutation_rate=MUTATION_RATE):
    """Adds noise to the genes with a given probability."""
    mutated = np.copy(individual)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Add Gaussian noise (mean 0, standard deviation 0.5)
            mutated[i] += np.random.normal(0, 0.5)
            # Keep weights within a reasonable limit (e.g. -5 to 5)
            mutated[i] = np.clip(mutated[i], -5, 5)
    return mutated


def run_evolution(env, selection_method="tournament"):
    """Runs the genetic algorithm evolution.

    Args:
        env: Gymnasium environment (render_mode=None for training).
        selection_method: "tournament" or "rank".
    """
    print(f"Starting Evolution... [selection: {selection_method}]")
    population = create_population(POPULATION_SIZE, DNA_SIZE)

    best_individual_overall = None
    best_fitness_overall = -float('inf')
    max_fitness_history = []
    avg_fitness_history = []

    for generation in range(GENERATIONS):
        fitnesses = [evaluate_fitness(ind, env) for ind in population]

        max_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / POPULATION_SIZE
        max_fitness_history.append(max_fitness)
        avg_fitness_history.append(avg_fitness)

        best_idx = fitnesses.index(max_fitness)
        if max_fitness > best_fitness_overall:
            best_fitness_overall = max_fitness
            best_individual_overall = np.copy(population[best_idx])

        print(f"Generation {generation + 1}/{GENERATIONS} | Max Fit: {max_fitness:.3f} | Avg Fit: {avg_fitness:.3f}")

        new_population = [population[best_idx]]  # elitism

        while len(new_population) < POPULATION_SIZE:
            if selection_method == "rank":
                parent1 = rank_selection(population, fitnesses)
                parent2 = rank_selection(population, fitnesses)
            else:
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)

            child = mutate(crossover(parent1, parent2))
            new_population.append(child)

        population = new_population

    return best_individual_overall, best_fitness_overall, max_fitness_history, avg_fitness_history
