import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
from config import *



# --- Actor-Critic RNN Network with Multiple Actor Heads ---
class ActorCriticRNN(nn.Module):
    def __init__(self, obs_dim, action_dim_large, action_dim_small, hidden_size):
        super(ActorCriticRNN, self).__init__()
        self.hidden_size = hidden_size
        self.action_dim_large = action_dim_large
        self.action_dim_small = action_dim_small

        self.lstm = nn.LSTM(obs_dim, hidden_size, batch_first=True)
        self.shared_fc = nn.Linear(hidden_size, hidden_size // 2)

        # Actor heads
        self.actor_head_large = nn.Linear(hidden_size // 2, action_dim_large)
        self.actor_head_small = nn.Linear(hidden_size // 2, action_dim_small)

        # Critic head
        self.critic_head = nn.Linear(hidden_size // 2, 1)
    
    def update_lstm(self, obs, hidden_state):
        _, next_hidden_state = self.lstm(obs, hidden_state)

        return next_hidden_state        

    def forward(self, obs, hidden_state):
        lstm_out, next_hidden_state = self.lstm(obs, hidden_state)
        shared_features = F.relu(self.shared_fc(lstm_out))

        logits_large = self.actor_head_large(shared_features)
        logits_small = self.actor_head_small(shared_features)
        value = self.critic_head(shared_features)

        return (logits_large, logits_small), value, next_hidden_state

    def get_initial_hidden_state(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(DEVICE)
        return (h0, c0)

# --- Rollout Buffer for Varying Action Spaces ---
class RolloutBufferRNN:
    def __init__(self, obs_dim, hidden_size, rollout_steps, sequence_length, gamma, gae_lambda):
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.rollout_steps = rollout_steps
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_steps,), dtype=np.int64) # Action is single int
        self.round_types = np.empty((rollout_steps,), dtype=object) # Store "large" or "small"
        self.log_probs = np.zeros((rollout_steps,), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps,), dtype=np.float32)
        self.dones = np.zeros((rollout_steps,), dtype=np.bool_)
        self.values = np.zeros((rollout_steps,), dtype=np.float32)
        
        self.h_ins = np.zeros((rollout_steps, 1, hidden_size), dtype=np.float32)
        self.c_ins = np.zeros((rollout_steps, 1, hidden_size), dtype=np.float32)

        self.advantages = np.zeros((rollout_steps,), dtype=np.float32)
        self.returns = np.zeros((rollout_steps,), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, log_prob, value, h_in, c_in, round_type):
        assert self.ptr < self.rollout_steps
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.round_types[self.ptr] = round_type # Store round type
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.h_ins[self.ptr] = h_in.cpu().numpy()
        self.c_ins[self.ptr] = c_in.cpu().numpy()
        self.ptr += 1
    
    def compute_gae_and_returns(self, last_value, last_done): # Same as before
        advantage = 0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * advantage
            self.advantages[t] = advantage
        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)



    def get_sequences(self, batch_size):
        num_samples = self.rollout_steps - self.sequence_length + 1
        if num_samples <= 0: return []

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) == 0: continue

            seq_obs, seq_actions, seq_log_probs, seq_advantages, seq_returns = [], [], [], [], []
            seq_h_initial, seq_c_initial = [], []
            # seq_round_types_initial = [] # Store the round type for the *start* of the sequence OLD
            seq_all_round_types_for_sequence = []
            for start_idx in batch_indices:
                end_idx = start_idx + self.sequence_length
                seq_obs.append(self.observations[start_idx:end_idx])
                seq_actions.append(self.actions[start_idx:end_idx])
                seq_log_probs.append(self.log_probs[start_idx:end_idx])
                seq_advantages.append(self.advantages[start_idx:end_idx])
                seq_returns.append(self.returns[start_idx:end_idx])
                seq_h_initial.append(self.h_ins[start_idx])
                seq_c_initial.append(self.c_ins[start_idx])
                # We assume the round_type for the first step of sequence applies to the whole training sequence
                # This is a simplification. If round_type can change *within* a sequence_length,
                # the update logic would need to be more granular.
                # seq_round_types_initial.append(self.round_types[start_idx]) Old
                seq_all_round_types_for_sequence.append(self.round_types[start_idx:end_idx]) # NEW

            yield (
                torch.tensor(np.array(seq_obs), dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(seq_actions), dtype=torch.int64).to(DEVICE),
                torch.tensor(np.array(seq_log_probs), dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(seq_advantages), dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(seq_returns), dtype=torch.float32).to(DEVICE),
                (torch.tensor(np.array(seq_h_initial), dtype=torch.float32).transpose(0,1).to(DEVICE),
                 torch.tensor(np.array(seq_c_initial), dtype=torch.float32).transpose(0,1).to(DEVICE)),
                #np.array(seq_round_types_initial) # batch_size array of strin
                np.array(seq_all_round_types_for_sequence) # Shape: (minibatch_size, sequence_length) of strings
            )
    def clear(self): self.ptr = 0


# --- PPO Agent for Varying Action Spaces ---
class PPO_RNN_Agent:
    def __init__(self, obs_dim, action_dim_large, action_dim_small, hidden_size, lr, gamma, gae_lambda,
                 ppo_clip_eps, ppo_epochs, minibatch_size, sequence_length,
                 entropy_coef, value_coef, max_grad_norm, rollout_steps):
        self.policy = ActorCriticRNN(obs_dim, action_dim_large, action_dim_small, hidden_size).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.action_dim_large = action_dim_large
        self.action_dim_small = action_dim_small
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip_eps = ppo_clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.sequence_length = sequence_length
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.buffer = RolloutBufferRNN(obs_dim, hidden_size, rollout_steps, sequence_length, gamma, gae_lambda)

    def select_action(self, obs, hidden_state, current_round_type, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE) # (1, 1, obs_dim)
        
        with torch.no_grad():
            (logits_large, logits_small), value, next_hidden_state = self.policy(obs_tensor, hidden_state)
            # logits_large/small: (1, 1, action_dim_*)

        if current_round_type == "large":
            dist = Categorical(logits=logits_large.squeeze(1))
        elif current_round_type == "small":
            dist = Categorical(logits=logits_small.squeeze(1))
        else:
            raise ValueError(f"Unknown round_type: {current_round_type}")

        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item(), next_hidden_state

    def update(self):
        self.policy.train()

        for epoch_num in range(self.ppo_epochs):
            # ... (data loading) ...
            for i, (batch_obs, batch_actions, batch_old_log_probs, \
                batch_advantages, batch_returns, batch_initial_hidden, \
                batch_seq_round_types) in enumerate(self.buffer.get_sequences(self.minibatch_size)):
                
                (new_logits_large_seq, new_logits_small_seq), new_values_seq, _ = \
                    self.policy(batch_obs, batch_initial_hidden)
                new_values_seq = new_values_seq.squeeze(-1)

                # Lists to hold per-sequence tensors
                # Each element in these lists will be a tensor of shape (sequence_length,)
                log_probs_sequences_list = []
                entropy_sequences_list = []

                for b in range(batch_obs.size(0)): # Iterate over sequences in minibatch
                    # Lists to hold per-step tensors for the current sequence `b`
                    log_probs_steps_for_seq_b = []
                    entropy_steps_for_seq_b = []

                    for t in range(self.sequence_length): # Iterate over timesteps in sequence
                        current_step_round_type = batch_seq_round_types[b, t]
                        current_step_action = batch_actions[b, t]

                        if current_step_round_type == "large":
                            step_logits = new_logits_large_seq[b, t, :]
                        elif current_step_round_type == "small":
                            step_logits = new_logits_small_seq[b, t, :]
                        else:
                            raise ValueError(f"Unknown round_type '{current_step_round_type}' in batch step b={b}, t={t}")
                        
                        dist = Categorical(logits=step_logits)
                        log_probs_steps_for_seq_b.append(dist.log_prob(current_step_action))
                        entropy_steps_for_seq_b.append(dist.entropy())
                    
                    # Stack the per-step tensors for sequence `b`
                    log_probs_sequences_list.append(torch.stack(log_probs_steps_for_seq_b)) # Shape: (sequence_length,)
                    entropy_sequences_list.append(torch.stack(entropy_steps_for_seq_b))   # Shape: (sequence_length,)
                
                # Stack the per-sequence tensors to form the batch
                new_log_probs = torch.stack(log_probs_sequences_list) # Shape: (minibatch_size, sequence_length)
                entropies = torch.stack(entropy_sequences_list)     # Shape: (minibatch_size, sequence_length)
                mean_entropy = entropies.mean() # Mean over batch and sequence

                # --- Sanity checks after stacking ---
                if not new_log_probs.requires_grad: print(f"ERROR AFTER STACK: new_log_probs no grad. grad_fn: {new_log_probs.grad_fn}")
                if not mean_entropy.requires_grad: print(f"ERROR AFTER STACK: mean_entropy no grad. grad_fn: {mean_entropy.grad_fn}")
                if not new_values_seq.requires_grad: print(f"ERROR: new_values_seq no grad. grad_fn: {new_values_seq.grad_fn}")

                # ... PPO Objective Calculation ...
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values_seq, batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * mean_entropy
                #loss = policy_loss + self.value_coef * value_loss                


                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        print(f"Mean entropy: {mean_entropy}")
        
        self.buffer.clear()
        return mean_entropy.item()

