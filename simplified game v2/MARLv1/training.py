import sys
sys.path.append('../GameModel')
from TempleOfHorror import TempleOfHorror

#Hyperparameters
NUMBER_OF_EPISODES = 1


for episode in range(NUMBER_OF_EPISODES):
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
        print(env.action_spaces)
        agent_key = env.action_spaces[f"agent_{agent_key}"][0]

        # Take the action in the environment
        done, next_state, reward,  info = env.step(agent_key)

        # Update the state
        state = next_state
    obs = env.create_observation(agent_key, state)
    print(state)












