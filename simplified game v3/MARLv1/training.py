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
count_wins_defender = 0
average_round = 0
obs_dim = [1,24]

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

total_timesteps = 100
episode_num = 0
all_episode_rewards = []

state = env.reset()
new_round = True
current_h, current_c = agent.policy.get_initial_hidden_state(batch_size=1)

# --- Training Loop ---
while total_timesteps < tqdm(range(total_timesteps)):   
    # Iterate over learning agents by rollout steps as in CFR.
    for learning_agent in ["attacker", "defender"]:
        for step in range(ROLLOUT_STEPS):
            h_in_for_buffer = current_h.clone()
            c_in_for_buffer = current_c.clone()

            # Message stage
            if new_round:
                new_round = False
                obs = env.create_observation(agent[-1], state)
                action, log_prob, value, (next_h, next_c) = agent.select_action(obs.reshape(1, -1), (current_h, current_c), "large")
                reward = 0 
                # choose a message
                state = env.step_message(action)

                next_obs = env.create_observation(agent_key, state)

                agent.buffer.add(obs, action, reward, done, log_prob, value, h_in_for_buffer, c_in_for_buffer, "large")

                obs = next_obs
                current_h, current_c = next_h, next_c

                episode_reward += reward
                total_timesteps += 1                


            # Choose an action (for now, we will choose a random action)

            obs = env.create_observation(agent_key, state)
            action, log_prob, value, (next_h, next_c) = agent.select_action(obs.reshape(1, -1), (current_h, current_c), "small")


            # Take the action in the environment
            done, next_state, reward,  info, winner = env.step(action)
            next_obs = env.create_observation(agent_key, state)   

            agent.buffer.add(obs, action, reward, done, log_prob, value, h_in_for_buffer, c_in_for_buffer, "small")
            
            # Update the state
            state = next_state
            obs = next_obs
            current_h, current_c = next_h, next_c   

            episode_reward += reward
            total_timesteps += 1

            if env.turns == 0:
                print("New round", env.round)
                new_round = True
        
            if done:
                state = env.reset()
                new_round = True
                current_h, current_c = agent.policy.get_initial_hidden_state(batch_size=1)
                if episode_num % 10 == 0:
                    avg_rew = np.mean(all_episode_rewards[-20:]) if len(all_episode_rewards) > 0 else 0.0
                    print(f"Ep: {episode_num}, Steps: {total_timesteps}, AvgRew: {avg_rew:.2f}, EpRew: {episode_reward:.2f}")
                episode_num += 1
                all_episode_rewards.append(episode_reward)
                episode_reward = 0

        with torch.no_grad():
            obs_tensor = torch.tensor(obs.reshape(1,1,-1), dtype=torch.float32).to(DEVICE)
            _, last_value, _ = agent.policy(obs_tensor, (current_h, current_c))
            last_value = last_value.item()
        
        agent.buffer.compute_gae_and_returns(last_value, done) 
        agent.update()

    print("Training finished.")
    
plt.figure(figsize=(12,6))
plt.plot(all_episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Rewards (Varying Action Space)")
N = 20
if len(all_episode_rewards) >= N:
    smoothed_rewards = np.convolve(all_episode_rewards, np.ones(N)/N, mode='valid')
    plt.plot(np.arange(N-1, len(all_episode_rewards)), smoothed_rewards, label=f'Smoothed (window {N})', color='red')
plt.legend()
plt.show()


# --- Training Loop ---
def train():
    env = TempleOfHorror()
    obs_dim = env.observation_space.shape[0]

    agent = PPO_RNN_Agent(
        obs_dim=obs_dim, action_dim_large=ACTION_DIM_LARGE, action_dim_small=ACTION_DIM_SMALL,
        hidden_size=HIDDEN_SIZE, lr=LR, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
        ppo_clip_eps=PPO_CLIP_EPS, ppo_epochs=PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH, entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF,
        max_grad_norm=MAX_GRAD_NORM, rollout_steps=ROLLOUT_STEPS
    )

    total_timesteps = 0
    episode_num = 0
    all_episode_rewards = []
    
    obs, info = env.reset()
    current_round_type = info["round_type"]
    current_h, current_c = agent.policy.get_initial_hidden_state(batch_size=1)
    
    episode_reward = 0

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps with varying action space env...")
    while total_timesteps < TOTAL_TIMESTEPS:
        for step in range(ROLLOUT_STEPS):
            h_in_for_buffer = current_h.clone()
            c_in_for_buffer = current_c.clone()

            action, log_prob, value, (next_h, next_c) = agent.select_action(
                obs.reshape(1, -1), (current_h, current_c), current_round_type
            )
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_round_type = info["round_type"] # Get round type for next step (usually same within episode)
            
            agent.buffer.add(obs, action, reward, done, log_prob, value, 
                             h_in_for_buffer, c_in_for_buffer, current_round_type)

            obs = next_obs
            current_h, current_c = next_h, next_c
            current_round_type = next_round_type # Update for the next iteration
            
            episode_reward += reward
            total_timesteps += 1

            if done:
                obs, info = env.reset()
                current_round_type = info["round_type"]
                current_h, current_c = agent.policy.get_initial_hidden_state(batch_size=1)
                all_episode_rewards.append(episode_reward)
                if episode_num % 10 == 0:
                    avg_rew = np.mean(all_episode_rewards[-20:]) if len(all_episode_rewards) > 0 else 0.0
                    print(f"Ep: {episode_num}, Steps: {total_timesteps}, AvgRew: {avg_rew:.2f}, EpRew: {episode_reward:.2f}, Round: {current_round_type}")
                episode_reward = 0
                episode_num += 1
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs.reshape(1,1,-1), dtype=torch.float32).to(DEVICE)
            _, last_value, _ = agent.policy(obs_tensor, (current_h, current_c))
            last_value = last_value.item()
        
        agent.buffer.compute_gae_and_returns(last_value, done) 
        agent.update()

    env.close()
    print("Training finished.")
    
    plt.figure(figsize=(12,6))
    plt.plot(all_episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards (Varying Action Space)")
    N = 20
    if len(all_episode_rewards) >= N:
        smoothed_rewards = np.convolve(all_episode_rewards, np.ones(N)/N, mode='valid')
        plt.plot(np.arange(N-1, len(all_episode_rewards)), smoothed_rewards, label=f'Smoothed (window {N})', color='red')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()










