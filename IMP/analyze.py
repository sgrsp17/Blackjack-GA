import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import run_evolution, evaluate_fitness, GENERATIONS

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


def _save_crossover_plot(results, selection, methods, colors, filename):
    plt.figure(figsize=(10, 6))
    generations = range(1, GENERATIONS + 1)
    for method, color in zip(methods, colors):
        max_hist, avg_hist = results[selection][method]
        plt.plot(generations, max_hist, label=f'Max Fit ({method})', color=color, linewidth=2)
        plt.plot(generations, avg_hist, label=f'Avg Fit ({method})', color=color, linestyle='--', alpha=0.7)
    plt.title(f'Impact of Crossover Methods on Evolution ({selection.capitalize()}, Mutation=0.1)')
    plt.xlabel('Generations')
    plt.ylabel('Fitness (Average Reward)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()


def analyze_crossover_methods():
    print("\n--- Starting Crossover Method Analysis ---")
    methods = ["uniform", "single_point", "multi_point"]
    selections = ["tournament", "rank"]
    colors_per_selection = {
        "tournament": ['firebrick', 'darkorange', 'gold'],
        "rank":       ['navy', 'steelblue', 'teal'],
    }

    results = {s: {} for s in selections}
    env = gym.make('Blackjack-v1', render_mode=None)

    for selection in selections:
        for method in methods:
            print(f"\nEvaluating Crossover Method: {method} | Selection: {selection}")
            _, _, max_hist, avg_hist, _ = run_evolution(
                env,
                selection_method=selection,
                crossover_method=method,
                mutation_rate=0.1
            )
            results[selection][method] = (max_hist, avg_hist)

    env.close()

    # individual plots per selection method
    for selection in selections:
        _save_crossover_plot(
            results, selection, methods, colors_per_selection[selection],
            f'crossover_analysis_{selection}.png'
        )

    # combined plot
    plt.figure(figsize=(12, 6))
    generations = range(1, GENERATIONS + 1)
    for selection in selections:
        for method, color in zip(methods, colors_per_selection[selection]):
            max_hist, avg_hist = results[selection][method]
            plt.plot(generations, max_hist, label=f'Max Fit ({selection}/{method})', color=color, linewidth=2)
            plt.plot(generations, avg_hist, label=f'Avg Fit ({selection}/{method})', color=color, linestyle='--', alpha=0.7)
    plt.title('Impact of Crossover Methods on Evolution (Tournament vs Rank, Mutation=0.1)')
    plt.xlabel('Generations')
    plt.ylabel('Fitness (Average Reward)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=7)
    plt.savefig('crossover_analysis.png')
    print("Saved crossover_analysis.png")
    plt.close()


def compare_best_combinations(n_eval_games=500, n_repeats=5):
    """Grid search over selection/crossover/mutation combinations.

    Each combination is trained n_repeats times; the best individual of each
    run is evaluated over n_eval_games. The plot shows mean win rate with
    standard deviation error bars across repeats.
    """
    print(f"\n--- Grid Search: Best Parameter Combination ({n_repeats} repeats) ---")
    selections    = ["tournament", "rank"]
    crossovers    = ["uniform", "single_point", "multi_point"]
    mutation_rates = [0.01, 0.1]

    env_train = gym.make('Blackjack-v1', render_mode=None)
    env_eval  = gym.make('Blackjack-v1', render_mode=None)

    labels = []
    means  = []
    stds   = []
    reward_means = []

    for selection in selections:
        for crossover in crossovers:
            for rate in mutation_rates:
                label = f"{selection}\n{crossover}\nrate={rate}"
                print(f"\nTraining: {label.replace(chr(10), ' | ')}")
                run_win_rates = []
                run_avg_rewards = []
                for rep in range(n_repeats):
                    best_individual, _, _, _, _ = run_evolution(
                        env_train,
                        selection_method=selection,
                        crossover_method=crossover,
                        mutation_rate=rate,
                    )
                    avg_reward, win_rate = evaluate_fitness(
                        best_individual, env_eval, n_games=n_eval_games, return_stats=True
                    )
                    run_win_rates.append(win_rate)
                    run_avg_rewards.append(avg_reward)
                    print(f"  Repeat {rep + 1}/{n_repeats}: win rate {win_rate:.2%} (avg reward: {avg_reward:.4f})")
                mean_wr = np.mean(run_win_rates)
                std_wr = np.std(run_win_rates)
                mean_ar = np.mean(run_avg_rewards)
                print(f"  => Mean win rate: {mean_wr:.2%} (+/- {std_wr:.2%}) | Mean avg reward: {mean_ar:.4f}")
                labels.append(label)
                means.append(mean_wr)
                stds.append(std_wr)
                reward_means.append(mean_ar)

    env_train.close()
    env_eval.close()

    # sort by mean win rate descending
    sorted_rows = sorted(zip(means, stds, reward_means, labels), reverse=True)
    means_sorted, stds_sorted, rewards_sorted, labels_sorted = zip(*sorted_rows)

    colors = ['gold' if i == 0 else 'steelblue' for i in range(len(labels_sorted))]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(labels_sorted)), means_sorted, yerr=stds_sorted,
                   color=colors, edgecolor='black', linewidth=0.5,
                   capsize=4, error_kw={'alpha': 0.7})
    plt.xticks(range(len(labels_sorted)), labels_sorted, fontsize=8)
    plt.ylabel(f'Win Rate (mean of {n_repeats} runs, {n_eval_games} games each)')
    plt.title(f'Best Parameter Combination — Grid Search ({n_repeats} repeats x {n_eval_games}-game evaluation)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar, mean, std, reward in zip(bars, means_sorted, stds_sorted, rewards_sorted):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.003,
                 f'{mean:.1%}\nAR={reward:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig('best_combination.png')
    print("\nSaved best_combination.png")
    plt.close()

    print("\n--- Ranking (by mean win rate) ---")
    for i, (mean, std, reward, lbl) in enumerate(zip(means_sorted, stds_sorted, rewards_sorted, labels_sorted), 1):
        print(f"  {i:2}. {lbl.replace(chr(10), ' | '):<45} win rate: {mean:.2%} (+/- {std:.2%}) | avg reward: {reward:.4f}")


if __name__ == "__main__":
    analyze_mutation_rates()
    analyze_crossover_methods()
    compare_best_combinations()
    print("\nAnalysis complete! Check 'mutation_analysis.png', 'crossover_analysis_tournament.png', 'crossover_analysis_rank.png', 'crossover_analysis.png' and 'best_combination.png'.")

