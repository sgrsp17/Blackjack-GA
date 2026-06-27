import numpy as np
import gymnasium as gym
from genetic_algorithm import evaluate_fitness

def test_saved_agent():
    print("Loading the best agent from 'best_agent.npy'...")
    try:
        best_weights = np.load('best_agent.npy')
    except FileNotFoundError:
        print("\n[ERROR] File 'best_agent.npy' not found!")
        print("Please run 'main.py' first to train and save the agent.")
        return

    print("Agent loaded successfully! Starting visual simulation of 5 games...")
    
    try:
        env_visual = gym.make('Blackjack-v1', render_mode="human")
        evaluate_fitness(best_weights, env_visual, n_games=5, verbose=True, agent_type="thinking")
        env_visual.close()
    except Exception as e:
        print(f"\n[WARNING] Failed to open the visual window: {e}")
        print("Make sure pygame is installed correctly.")

if __name__ == "__main__":
    test_saved_agent()
