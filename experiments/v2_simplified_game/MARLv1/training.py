import sys
sys.path.append('../GameModel')
from TempleOfHorror import TempleOfHorror
import numpy as np
from tqdm import tqdm
from sys import argv
#Hyperparameters
NUMBER_EPISODES = int(argv[1])
count_wins_defender = 0
for episode in tqdm(range(NUMBER_EPISODES)):
    # Create a new environment
    env = TempleOfHorror()

    # Reset the environment to start a new episode
    state = env.reset()


    for agent in env.agents:
        obs = env.create_observation(agent[-1], state)
        # choose a message
        state = env.step_message((1,1))

    agent_key = "0"
    while not env.done:
        # Choose an action (for now, we will choose a random action)

        obs = env.create_observation(agent_key, state)
        print(state)
        #print(env.action_spaces)
        agent_key = np.random.choice(env.action_spaces[f"agent_{agent_key}"])

        # Take the action in the environment
        done, next_state, reward,  info, winner = env.step(agent_key)

        # Update the state
        state = next_state
    obs = env.create_observation(agent_key, state)
    if winner == 1:
        count_wins_defender += 1

print("Defender wins: ", count_wins_defender/NUMBER_EPISODES)
print("Attacker wins: ", (NUMBER_EPISODES - count_wins_defender)/NUMBER_EPISODES)














