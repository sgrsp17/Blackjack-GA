import numpy as np
import gymnasium as gym
from genetic_algorithm import run_evolution, evaluate_fitness, GENERATIONS
from visualization import plot_fitness_history, EvolutionDisplay, show_evolution_summary

SELECTION_METHOD = "tournament"  # "tournament" or "rank"


def main():
    """Orchestrates the neuroevolution training and best-agent playback."""
    env = gym.make('Blackjack-v1', render_mode=None)

    display = EvolutionDisplay(GENERATIONS, SELECTION_METHOD)
    best_individual, best_fitness, max_fitness_history, avg_fitness_history, total_time = run_evolution(
        env, selection_method=SELECTION_METHOD, progress_callback=display.update
    )
    display.close()
    env.close()

    print("\nEvolution Completed!")
    print(f"Best Overall Fitness: {best_fitness:.3f}")
    print(f"Total Training Time:  {total_time:.1f}s")

    np.save('best_agent.npy', best_individual)
    print("Best agent weights saved to 'best_agent.npy'!")

    plot_fitness_history(max_fitness_history, avg_fitness_history)

    show_evolution_summary(best_fitness, total_time, SELECTION_METHOD, GENERATIONS)

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