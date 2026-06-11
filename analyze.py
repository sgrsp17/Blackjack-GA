import gymnasium as gym
import matplotlib.pyplot as plt
from genetic_algorithm import run_evolution, GENERATIONS

def analyze_mutation_rates():
    print("\n--- Starting Mutation Rate Analysis ---")
    rates = [0.01, 0.1, 0.5, 0.8]
    colors = ['blue', 'green', 'orange', 'red']
    
    plt.figure(figsize=(10, 6))
    env = gym.make('Blackjack-v1', render_mode=None)
    
    for rate, color in zip(rates, colors):
        print(f"\nEvaluating Mutation Rate: {rate}")
        _, _, max_hist, avg_hist, _ = run_evolution(
            env, 
            selection_method="tournament", 
            crossover_method="uniform", 
            mutation_rate=rate
        )
        generations = range(1, GENERATIONS + 1)
        plt.plot(generations, max_hist, label=f'Max Fit (Rate={rate})', color=color, linewidth=2)
        plt.plot(generations, avg_hist, label=f'Avg Fit (Rate={rate})', color=color, linestyle='--', alpha=0.7)

    env.close()
    
    plt.title('Impact of Mutation Rate on Evolution (Tournament, Uniform Crossover)')
    plt.xlabel('Generations')
    plt.ylabel('Fitness (Average Reward)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.savefig('mutation_analysis.png')
    print("Saved mutation_analysis.png")
    plt.close()


def analyze_crossover_methods():
    print("\n--- Starting Crossover Method Analysis ---")
    methods = ["uniform", "single_point", "multi_point"]
    selections = ["tournament", "rank"]
    colors = {
        "tournament": ['purple', 'cyan', 'magenta'],
        "rank":       ['orange', 'blue', 'green'],
    }

    plt.figure(figsize=(12, 6))
    env = gym.make('Blackjack-v1', render_mode=None)

    for selection in selections:
        for method, color in zip(methods, colors[selection]):
            print(f"\nEvaluating Crossover Method: {method} | Selection: {selection}")
            _, _, max_hist, avg_hist, _ = run_evolution(
                env,
                selection_method=selection,
                crossover_method=method,
                mutation_rate=0.1
            )
            generations = range(1, GENERATIONS + 1)
            plt.plot(generations, max_hist, label=f'Max Fit ({selection}/{method})', color=color, linewidth=2)
            plt.plot(generations, avg_hist, label=f'Avg Fit ({selection}/{method})', color=color, linestyle='--', alpha=0.7)

    env.close()

    plt.title('Impact of Crossover Methods on Evolution (Tournament vs Rank, Mutation=0.1)')
    plt.xlabel('Generations')
    plt.ylabel('Fitness (Average Reward)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=7)
    plt.savefig('crossover_analysis.png')
    print("Saved crossover_analysis.png")
    plt.close()

if __name__ == "__main__":
    analyze_mutation_rates()
    analyze_crossover_methods()
    print("\nAnalysis complete! Check 'mutation_analysis.png' and 'crossover_analysis.png'.")
