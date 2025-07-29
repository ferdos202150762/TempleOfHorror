import sys
sys.path.append('../GameModel')
from TempleOfHorror import TempleOfHorror
from PPO_agent import PPO_RNN_Agent
from config import *
import tqdm 
import numpy as np
import torch


obs_dim = 25
# ... agent initialization ...
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

# Option to load pre-trained models
LOAD_PRETRAINED = True # Set to False to train from scratch
if LOAD_PRETRAINED:
    agent_attacker = torch.load("attacker_agent_final_4.pth") # or a specific checkpoint
    agent_defender = torch.load("defender_agent_final_4.pth") # or a specific checkpoint

agents = {
    "attacker": agent_attacker,
    "defender": agent_defender
}
# ... rest of your setup ...

def test_agents(env, agents_dict, num_test_episodes=100):
    print("\n--- Starting Testing Phase ---")
    
    # Ensure agents are in evaluation mode (disables dropout/batchnorm if any)
    # and use deterministic actions
    agents_dict["attacker"].policy.eval()
    agents_dict["defender"].policy.eval()

    total_rewards_attacker = []
    total_rewards_defender = []
    wins_attacker = 0
    wins_defender = 0
    draws = 0 # If your game has draws

    for episode in range(num_test_episodes):
        state = env.reset()
        acting_player = env.player_role['agent_0']
        print(env.player_role)
        # Get initial hidden states for both agents for this episode
        # Important: Reset hidden states at the start of each test episode
        hidden_states = {
            "attacker": agents_dict["attacker"].policy.get_initial_hidden_state(batch_size=1),
            "defender": agents_dict["defender"].policy.get_initial_hidden_state(batch_size=1)
        }
        
        episode_rewards_map = {"attacker": 0, "defender": 0} # Assuming reward is per player
        done = False
        current_step = 0
        max_episode_steps_test = 100 # Or whatever is appropriate for your game
        while current_step < max_episode_steps_test:
            while not done:

                if not env.message_provided and env.provide_message == 0:
                    # Reset acting agent when messages start again
                    acting_player = env.player_role['agent_0']

                agent_role = acting_player[:-2] # "attacker" or "defender"
                current_agent_nn = agents_dict[agent_role]
                current_h, current_c = hidden_states[agent_role]

                agent_number = env.player_number[acting_player]
                obs = env.create_observation(agent_number, state)

                round_type = "large" if not env.message_provided else "small"

                action_index, _, _, (next_h, next_c) = current_agent_nn.select_action(
                    obs.reshape(1, -1), 
                    (current_h, current_c), 
                    round_type,
                    deterministic=False # Use deterministic actions for testing
                )
                
                
                # Update the hidden state for the current acting agent
                hidden_states[agent_role] = (next_h, next_c)
                print(acting_player)
                if round_type == "large": # Message stage

                    action = env.message_space[action_index]
                    next_state = env.step_message(action)
                    # No immediate reward for messages in your current setup for learning agent
                    # For testing, you might not track reward here or assume 0
                    acting_player = env.player_role[f"agent_{env.provide_message}"]




                else: # Action stage

                    action = env.action_spaces[f"agent_{agent_number}"][action_index]
                    
                    done, next_state, reward_signal, info, winner_info = env.step(action)
                    
                    # Accumulate rewards for both players if reward_signal is a dict/list
                    # Adjust this based on how your env.step() returns rewards
                    if isinstance(reward_signal, dict) or isinstance(reward_signal, list):
                        # Assuming reward_signal is a list [attacker_reward, defender_reward] based on agent_number
                        if agent_number == 0: # Attacker is agent_0
                            episode_rewards_map["attacker"] += reward_signal[0]
                            episode_rewards_map["defender"] += reward_signal[1]
                        else: # Defender is agent_1
                            episode_rewards_map["attacker"] += reward_signal[0] # or however it's structured
                            episode_rewards_map["defender"] += reward_signal[1]
                    else: # If reward is a single value for the acting player
                        episode_rewards_map[agent_role] += reward_signal

                    acting_player = env.player_role[f"agent_{action}"] # This line might need adjustment
                                                                    # based on how `action` relates to next player
                                                                    # Is `action` the player number or action value?
                                                                    # Assuming `action` here refers to the action value,
                                                                    # not the player index that made the choice.
                                                                    # Your training loop has `acting_player = env.player_role[f"agent_{action}"]`
                                                                    # This seems like `action` is expected to be the agent_number for the next turn
                                                                    # which is unusual. `env.step` usually determines the next player internally.
                                                                    # Let's assume env handles next player correctly:

                    print(acting_player)
                state = next_state

            current_step += 1
            
            if done:
                # Determine winner based on your environment's winner_info or other logic
                if winner_info == 0: # Example
                    wins_attacker += 1
                elif winner_info == 1:
                    wins_defender += 1
                elif winner_info == "draw": # Or if no winner and game ends
                    draws += 1
                break # Exit while loop for this episode

        total_rewards_attacker.append(episode_rewards_map["attacker"])
        total_rewards_defender.append(episode_rewards_map["defender"])

    avg_attacker_reward = np.mean(total_rewards_attacker)
    avg_defender_reward = np.mean(total_rewards_defender)
    std_attacker_reward = np.std(total_rewards_attacker)
    std_defender_reward = np.std(total_rewards_defender)    
    print(f"\n--- Test Results ({num_test_episodes} episodes) ---")
    print(f"Average Attacker Reward: {avg_attacker_reward:.2f}, Std: {std_attacker_reward:.2f}")
    print(f"Average Defender Reward: {avg_defender_reward:.2f}, Std: {std_defender_reward:.2f}")
    print(f"Attacker Wins: {wins_attacker} ({(wins_attacker/num_test_episodes)*100:.1f}%)")
    print(f"Defender Wins: {wins_defender} ({(wins_defender/num_test_episodes)*100:.1f}%)")
    if draws > 0:
        print(f"Draws: {draws} ({(draws/num_test_episodes)*100:.1f}%)")
    

    
    return avg_attacker_reward, avg_defender_reward, wins_attacker, wins_defender

if __name__ == "__main__":
    # ... (your existing setup and training code) ...
    
    # After training is complete, or if you just want to test pre-loaded models:
    # Ensure models are loaded if you're not continuing from training
    env = TempleOfHorror()

    test_agents(env, agents, num_test_episodes=100)