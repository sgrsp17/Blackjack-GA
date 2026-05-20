import gymnasium as gym
import time

# Create the environment
env = gym.make('Blackjack-v1', render_mode="human")

# Restart the game to get new inputs(player's total, dealer's total, dealer's ace)
# Observation will return something like (14, 10, FALSE)
observation,info=env.reset()
finished = False

while not finished:
    # Where GA would consult the individual's DNA to choose a decision
    action = 1  # HIT just to test   
     
    # Step the enviroment forward
    observation,reward,finished,truncated,info=env.step(action)
    print("Player's total:",observation[0],"Dealer's total:",observation[1],"Dealer's ace:",observation[2])
    print("Reward:",reward,"Finished:",finished,"Truncated:",truncated)    





# Esperar 3 segundos para conseguires ver o resultado final
time.sleep(3)
env.close() # Fechar a janela do jogo de forma limpa

print(env.observation_space)
