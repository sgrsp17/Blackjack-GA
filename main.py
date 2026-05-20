import numpy as np
import gymnasium as gym
import random
from neural_network import NeuralNetwork

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

def evaluate_fitness(individual, env, n_games=N_GAMES):
    """Evaluates an individual by playing N_GAMES and returning the average reward."""
    nn = NeuralNetwork(weights=individual)
    total_reward = 0.0

    for _ in range(n_games):
        observation, info = env.reset()
        finished = False
        truncated = False
        
        while not (finished or truncated):
            # The observation in blackjack is (player_sum, dealer_card, usable_ace)
            # The predict output should return 0 (Stand) or 1 (Hit)
            action = nn.predict(observation)
            
            # Ensure the action is a valid int (in case a float is returned)
            if action not in [0, 1]:
                action = 1 if action > 0.5 else 0
                
            observation, reward, finished, truncated, info = env.step(action)
            total_reward += reward

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

def main():
    # Create the silent environment for training
    env = gym.make('Blackjack-v1', render_mode=None)
    
    print("Starting Evolution...")
    population = create_population(POPULATION_SIZE, DNA_SIZE)
    
    best_individual_overall = None
    best_fitness_overall = -float('inf')

    for generation in range(GENERATIONS):
        # 1. Evaluate the population
        fitnesses = [evaluate_fitness(ind, env) for ind in population]
        
        max_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / POPULATION_SIZE
        
        # Save the all-time best
        best_idx = fitnesses.index(max_fitness)
        if max_fitness > best_fitness_overall:
            best_fitness_overall = max_fitness
            best_individual_overall = np.copy(population[best_idx])
            
        print(f"Generation {generation + 1}/{GENERATIONS} | Max Fit: {max_fitness:.3f} | Avg Fit: {avg_fitness:.3f}")
        
        # 2. Create the new generation
        new_population = []
        
        # Elitism: Pass the best of this generation directly to the next
        new_population.append(population[best_idx])
        
        while len(new_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child)
            
            new_population.append(child)
            
        population = new_population

    env.close()
    
    print("\nEvolution Completed!")
    print(f"Best Overall Fitness: {best_fitness_overall:.3f}")
    
    # Show the best individual playing
    print("\nTesting the best individual visually...")
    env_visual = gym.make('Blackjack-v1', render_mode="human")
    # Test for 15 games so we can observe
    evaluate_fitness(best_individual_overall, env_visual, n_games=15)
    env_visual.close()

if __name__ == "__main__":
    main()