import numpy as np
import gymnasium as gym
from genetic_algorithm import run_evolution, evaluate_fitness, GENERATIONS
from visualization import plot_fitness_history, plot_comparison, EvolutionDisplay, show_evolution_summary

SELECTION_METHOD = "tournament"  # "tournament" or "rank" — used when COMPARE_MODE is False
COMPARE_MODE = False             # True: runs both methods and plots a comparison graph


def run_method(method):
    env = gym.make('Blackjack-v1', render_mode=None)
    display = EvolutionDisplay(GENERATIONS, method)
    best_individual, best_fitness, max_hist, avg_hist, total_time = run_evolution(
        env, selection_method=method, progress_callback=display.update
    )
    display.close()
    env.close()
    print(f"\n[{method}] Best Fitness: {best_fitness:.3f} | Time: {total_time:.1f}s")
    return best_individual, best_fitness, max_hist, avg_hist, total_time


def main():
    """Orchestrates the neuroevolution training and best-agent playback."""
    if COMPARE_MODE:
        t_ind, t_fit, t_max, t_avg, t_time = run_method("tournament")
        r_ind, r_fit, r_max, r_avg, r_time = run_method("rank")

        plot_comparison(t_max, t_avg, r_max, r_avg)

        # Play with whichever method produced the best agent
        best_individual = t_ind if t_fit >= r_fit else r_ind
        best_fitness = max(t_fit, r_fit)
        best_method = "tournament" if t_fit >= r_fit else "rank"
        total_time = t_time + r_time

        np.save('best_agent.npy', best_individual)
        print(f"\nBest agent overall: {best_method} (fitness {best_fitness:.3f})")
        print("Saved to 'best_agent.npy'.")

        show_evolution_summary(best_fitness, total_time, f"comparison ({best_method} won)", GENERATIONS * 2)
    else:
        best_individual, best_fitness, max_hist, avg_hist, total_time = run_method(SELECTION_METHOD)

        np.save('best_agent.npy', best_individual)
        print("Best agent weights saved to 'best_agent.npy'!")

        plot_fitness_history(max_hist, avg_hist, SELECTION_METHOD)
        show_evolution_summary(best_fitness, total_time, SELECTION_METHOD, GENERATIONS)

    print("\n--- Playing the BEST TRAINED individual ---")
    try:
        env_visual = gym.make('Blackjack-v1', render_mode="human")
        evaluate_fitness(best_individual, env_visual, n_games=5, verbose=True, agent_type="thinking")
        env_visual.close()
    except Exception as e:
        print(f"\n[Warning] An error occurred while trying to open the visual Pygame window: {e}")

if __name__ == "__main__":
    main()
