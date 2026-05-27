import numpy as np
import gymnasium as gym
from genetic_algorithm import run_evolution, evaluate_fitness
from visualization import plot_fitness_history


def main():
    """Orchestrates the neuroevolution training and best-agent playback."""
    # Train the population
    env = gym.make('Blackjack-v1', render_mode=None)
    best_individual, best_fitness, max_fitness_history, avg_fitness_history, total_time = run_evolution(env)
    env.close()

    print("\nEvolution Completed!")
    print(f"Best Overall Fitness: {best_fitness:.3f}")
    print(f"Total Training Time:  {total_time:.1f}s")
    
    # Save the best agent
    np.save('best_agent.npy', best_individual)
    print("Best agent weights saved to 'best_agent.npy'!")
    
    # Plot the fitness evolution
    plot_fitness_history(max_fitness_history, avg_fitness_history)

    # Play with the best trained agent
    print("\n--- Playing the BEST TRAINED individual ---")
    try:
        env_visual = gym.make('Blackjack-v1', render_mode="human")
        evaluate_fitness(best_individual, env_visual, n_games=5, verbose=True, agent_type="thinking")
        env_visual.close()
    except Exception as e:
        print(f"\n[Warning] An error occurred while trying to open the visual Pygame window: {e}")
        print("This does not affect the neural network training, only the final visualization.")

if __name__ == "__main__":
    main()