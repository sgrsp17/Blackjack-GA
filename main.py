import numpy as np
import gymnasium as gym
import random
from neural_network import NeuralNetwork

# --- Constantes do Algoritmo Genético ---
POPULATION_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1
DNA_SIZE = 81  # 3*16 + 16 + 16*1 + 1 (Para 3 inputs, 16 hidden, 1 output)
N_GAMES = 100  # Jogos a jogar para avaliar cada indivíduo
TOURNAMENT_SIZE = 3

def create_population(size, dna_size):
    """Cria a população inicial com pesos aleatórios entre -1 e 1."""
    return [np.random.uniform(-1, 1, dna_size) for _ in range(size)]

def evaluate_fitness(individual, env, n_games=N_GAMES):
    """Avalia um indivíduo jogando N_GAMES e retornando a recompensa média."""
    nn = NeuralNetwork(weights=individual)
    total_reward = 0.0

    for _ in range(n_games):
        observation, info = env.reset()
        finished = False
        truncated = False
        
        while not (finished or truncated):
            # A observação no blackjack é (player_sum, dealer_card, usable_ace)
            # O output predict deve retornar 0 (Stand) ou 1 (Hit)
            action = nn.predict(observation)
            
            # Garantir que a ação é um int válido (caso devolvam um float)
            if action not in [0, 1]:
                action = 1 if action > 0.5 else 0
                
            observation, reward, finished, truncated, info = env.step(action)
            total_reward += reward

    return total_reward / n_games

def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    """Seleciona o melhor indivíduo entre K escolhidos aleatoriamente."""
    selected_indices = random.sample(range(len(population)), k)
    best_index = selected_indices[0]
    best_fitness = fitnesses[best_index]
    
    for idx in selected_indices[1:]:
        if fitnesses[idx] > best_fitness:
            best_fitness = fitnesses[idx]
            best_index = idx
            
    return population[best_index]

def crossover(parent1, parent2):
    """Uniform crossover: escolhe genes aleatoriamente dos dois pais."""
    child = np.zeros_like(parent1)
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child

def mutate(individual, mutation_rate=MUTATION_RATE):
    """Adiciona ruído aos genes com uma dada probabilidade."""
    mutated = np.copy(individual)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Adiciona ruído Gaussiano (média 0, desvio padrão 0.5)
            mutated[i] += np.random.normal(0, 0.5)
            # Manter os pesos num limite razoável (ex: -5 a 5)
            mutated[i] = np.clip(mutated[i], -5, 5)
    return mutated

def main():
    # Cria o ambiente silencioso para treino
    env = gym.make('Blackjack-v1', render_mode=None)
    
    print("Iniciando a Evolução...")
    population = create_population(POPULATION_SIZE, DNA_SIZE)
    
    best_individual_overall = None
    best_fitness_overall = -float('inf')

    for generation in range(GENERATIONS):
        # 1. Avaliar a população
        fitnesses = [evaluate_fitness(ind, env) for ind in population]
        
        max_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / POPULATION_SIZE
        
        # Guardar o melhor de sempre
        best_idx = fitnesses.index(max_fitness)
        if max_fitness > best_fitness_overall:
            best_fitness_overall = max_fitness
            best_individual_overall = np.copy(population[best_idx])
            
        print(f"Geração {generation + 1}/{GENERATIONS} | Max Fit: {max_fitness:.3f} | Avg Fit: {avg_fitness:.3f}")
        
        # 2. Criar a nova geração
        new_population = []
        
        # Elitismo: Passar o melhor desta geração diretamente para a próxima
        new_population.append(population[best_idx])
        
        while len(new_population) < POPULATION_SIZE:
            # Seleção
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutação
            child = mutate(child)
            
            new_population.append(child)
            
        population = new_population

    env.close()
    
    print("\nEvolução Concluída!")
    print(f"Melhor Fitness Global: {best_fitness_overall:.3f}")
    
    # Mostrar o melhor a jogar
    print("\nA testar o melhor indivíduo visualmente...")
    env_visual = gym.make('Blackjack-v1', render_mode="human")
    # Testa durante 5 jogos para podermos observar
    evaluate_fitness(best_individual_overall, env_visual, n_games=5)
    env_visual.close()

if __name__ == "__main__":
    main()