import sys
sys.path.append('../GameModel')
from TempleOfHorror import TempleOfHorror
import numpy as np
from tqdm import tqdm
from sys import argv
from config import *
from PPO_attacker import PPO_RNN_Agent
import torch
import matplotlib.pyplot as plt

#Hyperparameters
obs_dim = 25

env = TempleOfHorror()
agent_attacker = PPO_RNN_Agent(
    obs_dim=obs_dim, action_dim_large=ACTION_DIM_LARGE, action_dim_small=ACTION_DIM_SMALL,
    hidden_size=HIDDEN_SIZE, lr=LR, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
    ppo_clip_eps=PPO_CLIP_EPS, ppo_epochs=PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH, entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF,
    max_grad_norm=MAX_GRAD_NORM, rollout_steps=ROLLOUT_STEPS
)
agent_defender = PPO_RNN_Agent(
    obs_dim=obs_dim, action_dim_large=ACTION_DIM_LARGE, action_dim_small=ACTION_DIM_SMALL,
    hidden_size=HIDDEN_SIZE, lr=LR, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
    ppo_clip_eps=PPO_CLIP_EPS, ppo_epochs=PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE,
    sequence_length=SEQUENCE_LENGTH, entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF,
    max_grad_norm=MAX_GRAD_NORM, rollout_steps=ROLLOUT_STEPS
)
agents = {
    "attacker": agent_attacker,
    "defender": agent_defender
}

total_timesteps = 0 
episode_num = 0

all_episode_rewards = {'attacker': [], 'defender': []}

state = env.reset()
acting_player = env.player_role['agent_0']
agent = agents[acting_player[:-2]]

hidden_states_map = {
    "attacker": agents["attacker"].policy.get_initial_hidden_state(batch_size=1),
    "defender": agents["defender"].policy.get_initial_hidden_state(batch_size=1)
}


episode_reward = 0
# --- Training Loop ---
while total_timesteps < TOTAL_TIMESTEPS:   
    # Iterate over learning agents by rollout steps as in CFR.
    for learning_agent in ["attacker", "defender"]:
        learning_agent_nn = agents[learning_agent]
        for step in range(ROLLOUT_STEPS):
            if acting_player[:-2] == learning_agent: # Learning Agent
                current_h, current_c = hidden_states_map[learning_agent]

                agent_number = env.player_number[acting_player]
                h_in_for_buffer = current_h.clone()
                c_in_for_buffer = current_c.clone()

                obs = env.create_observation(agent_number, state)
                # Message stage
                if not env.message_provided: # If all messages have not been provided this agent provides a message

                    action_index, log_prob, value, (next_h, next_c) = learning_agent_nn.select_action(obs.reshape(1, -1), (current_h, current_c), "large")
                    reward = 0
                    done = False
                    # choose a message
                    action = env.message_space[action_index]
                    next_state = env.step_message(action)           
                    next_obs = env.create_observation(agent_number, next_state)  
                    learning_agent_nn.buffer.add(obs, action_index, reward, done, log_prob, value, h_in_for_buffer, c_in_for_buffer, "large")
                    episode_reward += reward
                    acting_player = env.player_role[f"agent_{env.provide_message}"]
                else:                 # Choose an action (for now, we will choose a random action)

                    action_index, log_prob, value, (next_h, next_c) = learning_agent_nn.select_action(obs.reshape(1, -1), (current_h, current_c), "small")
                    action = env.action_spaces[f"agent_{agent_number}"][action_index]
                    # Take the action in the environment
                    done, next_state, reward,  info, winner = env.step(action)           
                    next_obs = env.create_observation(agent_number, next_state)   
                    learning_agent_nn.buffer.add(obs, action_index, reward[agent_number], done, log_prob, value, h_in_for_buffer, c_in_for_buffer, "small")           
                    episode_reward += reward[agent_number]
                    acting_player = env.player_role[f"agent_{action}"] 

                all_episode_rewards[learning_agent].append(episode_reward)
            else: # Opponent Agent (not learning)
                current_h, current_c = hidden_states_map[acting_player[:-2]]
                opponent_agent = agents[acting_player[:-2]]
                agent_number = env.player_number[acting_player]
                h_in_for_buffer = current_h.clone()
                c_in_for_buffer = current_c.clone()
                obs = env.create_observation(agent_number, state)
                # Message stage
                if not env.message_provided:
                    action_index, log_prob, value, (next_h, next_c) = opponent_agent.select_action(obs.reshape(1, -1), (current_h, current_c), "large")
                    reward = 0 
                    action = env.message_space[action_index]
                    done = False
                    # choose a message
                    next_state = env.step_message(action) 
                    next_obs = env.create_observation(agent_number, next_state) 
                    acting_player = env.player_role[f"agent_{env.provide_message}"]  
                else:                 
                    action_index, log_prob, value, (next_h, next_c) = opponent_agent.select_action(obs.reshape(1, -1), (current_h, current_c), "small")
                    action = env.action_spaces[f"agent_{agent_number}"][action_index]

                    # Take the action in the environment
                    done, next_state, reward,  info, winner = env.step(action)                  
                    next_obs = env.create_observation(agent_number, next_state)               
                    acting_player = env.player_role[f"agent_{action}"]    
                                  
            # Update the state
            state = next_state
            obs = next_obs
            hidden_states_map[acting_player[:-2]] = (next_h, next_c) 
            total_timesteps += 1


            if done:
                print(env.round)
                state = env.reset()
                acting_player = env.player_role['agent_0']
                agent = agents[acting_player[:-2]]
                new_round = True
                hidden_states_map = {
                    "attacker": agents["attacker"].policy.get_initial_hidden_state(batch_size=1),
                    "defender": agents["defender"].policy.get_initial_hidden_state(batch_size=1)
                }
                if episode_num % 10 == 0:
                    avg_rew = np.mean(all_episode_rewards[learning_agent][-20:]) if len(all_episode_rewards[learning_agent]) > 0 else 0.0
                    print(f"Ep: {episode_num}, Steps: {total_timesteps}, AvgRew: {avg_rew:.2f}, EpRew: {episode_reward:.2f}")
                episode_reward = 0
                episode_num += 1


        with torch.no_grad():
            print("Update")
            obs_tensor = torch.tensor(obs.reshape(1,1,-1), dtype=torch.float32).to(DEVICE)
            _, last_value, _ = learning_agent_nn.policy(obs_tensor, (current_h, current_c))
            last_value = last_value.item()       
            learning_agent_nn.buffer.compute_gae_and_returns(last_value, done) 
            learning_agent_nn.update()

    print("Training finished.")

plt.figure(figsize=(14, 7)) # Adjusted figure size for two plots

# Colors for attacker and defender
colors = {'attacker': 'blue', 'defender': 'green'}
smooth_colors = {'attacker': 'lightblue', 'defender': 'lightgreen'} # Lighter for raw, darker for smooth
raw_plot_colors = {'attacker': 'lightblue', 'defender': 'lightgreen'}
smooth_plot_colors = {'attacker': 'blue', 'defender': 'green'}


num_episodes_attacker = len(all_episode_rewards['attacker'])
num_episodes_defender = len(all_episode_rewards['defender'])

# It's possible one agent has more recorded episodes if training stops mid-episode
# or if logging is slightly off. We'll plot up to the minimum length for aligned comparison.
min_episodes = min(num_episodes_attacker, num_episodes_defender)
if min_episodes == 0:
    print("No episode rewards recorded for one or both agents. Cannot plot.")
else:
    # Plot Attacker Rewards
    plt.plot(
        np.arange(min_episodes),
        all_episode_rewards['attacker'][:min_episodes],
        label='Attacker Raw Reward',
        color=raw_plot_colors['attacker'],
        alpha=0.6 # Make raw data a bit transparent
    )

    # Plot Defender Rewards
    plt.plot(
        np.arange(min_episodes),
        all_episode_rewards['defender'][:min_episodes],
        label='Defender Raw Reward',
        color=raw_plot_colors['defender'],
        alpha=0.6
    )

    # Smoothing Window
    N = 20  # You can adjust this

    # Smoothed Attacker Rewards
    if num_episodes_attacker >= N:
        smoothed_attacker = np.convolve(all_episode_rewards['attacker'], np.ones(N)/N, mode='valid')
        # Adjust x-axis for smoothed plot: starts after N-1 initial points
        plt.plot(
            np.arange(N - 1, num_episodes_attacker), # Use full length of attacker rewards for its smooth plot
            smoothed_attacker,
            label=f'Attacker Smoothed (window {N})',
            color=smooth_plot_colors['attacker'],
            linewidth=2
        )

    # Smoothed Defender Rewards
    if num_episodes_defender >= N:
        smoothed_defender = np.convolve(all_episode_rewards['defender'], np.ones(N)/N, mode='valid')
        plt.plot(
            np.arange(N - 1, num_episodes_defender), # Use full length of defender rewards for its smooth plot
            smoothed_defender,
            label=f'Defender Smoothed (window {N})',
            color=smooth_plot_colors['defender'],
            linewidth=2
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward per Episode")
    plt.title("Episode Rewards for Attacker and Defender")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()


# ... after the main training loop ...
print("Training finished.")
torch.save(agents["attacker"], "attacker_agent_final.pth")
torch.save(agents["defender"], "defender_agent_final.pth")
print(f"Total episodes trained: {episode_num}")
print(f"Final episode reward (might be incomplete): {episode_reward}")

# Plotting
# ... (your plotting code) ...