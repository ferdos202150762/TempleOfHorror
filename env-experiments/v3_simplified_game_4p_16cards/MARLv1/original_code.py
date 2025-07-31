Okay, this is a more complex and interesting scenario! It means the agent needs to adapt its output strategy based on some contextual information (the "round type" or current game phase).

Here's how we can approach this:

Environment Signal: The environment must somehow signal to the agent which action space is currently active. This could be:
Part of the observation (e.g., a specific feature indicates the round type).
A piece of information in the info dictionary returned by env.step() or env.reset().
A property of the environment object itself that the agent can query.
Policy Network Architecture:
The LSTM body and critic head can remain largely the same, as they process observations and estimate state values.
The actor part will need two separate output heads:
One head outputs logits for the 15-action space.
Another head outputs logits for the 2-action space.
During the forward pass, the network will compute logits for both heads.
Action Selection:
When selecting an action, the agent will first determine the current round_type.
Based on round_type, it will use the logits from the corresponding actor head to form a Categorical distribution and sample an action.
The log probability of the chosen action from the active head will be recorded.
Data Storage in Buffer:
The buffer needs to store not only the action taken but also the round_type active when that action was taken. This is crucial for the update step.
The action itself will be an integer, but its meaning (and the range it came from) depends on the round_type.
PPO Update Step:
When re-calculating log probabilities and entropy during the PPO update:
For each transition (or sequence step) in the minibatch, use the stored round_type to select the correct set of logits (from the large or small action head) from the policy network's current output.
Calculate the new log probability using the stored action and the selected logits.
Calculate entropy using the selected logits.
The PPO ratio exp(new_log_prob - old_log_prob) and subsequent loss calculations proceed as usual, but new_log_prob and entropy are now conditional on the round_type of that specific transition.
Let's modify the code. We'll need a dummy environment that exhibits this behavior.

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

# --- Hyperparameters ---
# ENV_NAME (will use dummy)
ACTION_DIM_LARGE = 15
ACTION_DIM_SMALL = 2
OBS_DIM_DUMMY = 4
HIDDEN_SIZE = 64
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 10
MINIBATCH_SIZE = 32
ROLLOUT_STEPS = 1024
SEQUENCE_LENGTH = 16
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

TOTAL_TIMESTEPS = 75_000 # Adjusted for potentially slower learning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Dummy Environment with Changing Action Spaces per Episode ---
class VaryingActionSpaceEnv(gym.Env):
    def __init__(self, obs_dim, action_dim_large, action_dim_small):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim_large = action_dim_large
        self.action_dim_small = action_dim_small
        
        # Observation space is fixed
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)
        
        # Action space will be set dynamically per episode
        self._action_space_large = spaces.Discrete(action_dim_large)
        self._action_space_small = spaces.Discrete(action_dim_small)
        self.action_space = self._action_space_large # Default
        
        self.current_round_type = "large" # "large" or "small"
        self.episode_step = 0
        self.max_episode_steps = 50 # Shorter episodes to see round changes

    def _set_round_type(self):
        # Simple rule: alternate every episode, or randomly
        if random.random() < 0.5: # Or use self.episode_count % 2 == 0
            self.current_round_type = "large"
            self.action_space = self._action_space_large
            # print("Switched to LARGE action space")
        else:
            self.current_round_type = "small"
            self.action_space = self._action_space_small
            # print("Switched to SMALL action space")

    def step(self, action):
        # Action validation depends on current_round_type
        if self.current_round_type == "large":
            assert self._action_space_large.contains(action), f"Invalid action {action} for large space"
        else:
            assert self._action_space_small.contains(action), f"Invalid action {action} for small space"

        self.episode_step += 1
        obs = self.observation_space.sample()
        
        # Dummy reward, slightly different based on round type to encourage learning
        if self.current_round_type == "large":
            reward = (action / (self.action_dim_large -1)) - 0.5 + np.random.rand() * 0.1
        else:
            reward = (action / (self.action_dim_small -1)) * 2 - 1.0 + np.random.rand() * 0.1
            
        terminated = self.episode_step >= self.max_episode_steps
        truncated = False 
        info = {"round_type": self.current_round_type}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        self._set_round_type() # Determine action space for this new episode
        obs = self.observation_space.sample()
        info = {"round_type": self.current_round_type}
        return obs, info

    def render(self): pass
    def close(self): pass

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
            seq_round_types_initial = [] # Store the round type for the *start* of the sequence

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
                seq_round_types_initial.append(self.round_types[start_idx])


            yield (
                torch.tensor(np.array(seq_obs), dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(seq_actions), dtype=torch.int64).to(DEVICE),
                torch.tensor(np.array(seq_log_probs), dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(seq_advantages), dtype=torch.float32).to(DEVICE),
                torch.tensor(np.array(seq_returns), dtype=torch.float32).to(DEVICE),
                (torch.tensor(np.array(seq_h_initial), dtype=torch.float32).transpose(0,1).to(DEVICE),
                 torch.tensor(np.array(seq_c_initial), dtype=torch.float32).transpose(0,1).to(DEVICE)),
                np.array(seq_round_types_initial) # batch_size array of strings
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
        for _ in range(self.ppo_epochs):
            for batch_obs, batch_actions, batch_old_log_probs, \
                batch_advantages, batch_returns, batch_initial_hidden, \
                batch_initial_round_types in self.buffer.get_sequences(self.minibatch_size):

                # batch_obs: (minibatch_size, sequence_length, obs_dim)
                # batch_actions: (minibatch_size, sequence_length)
                # batch_initial_round_types: (minibatch_size,) array of strings

                (new_logits_large_seq, new_logits_small_seq), new_values_seq, _ = \
                    self.policy(batch_obs, batch_initial_hidden)
                # new_logits_*: (minibatch_size, sequence_length, action_dim_*)
                new_values_seq = new_values_seq.squeeze(-1) # (minibatch_size, sequence_length)

                # Process each sequence in the batch according to its initial round type
                new_log_probs_list = []
                entropy_list = []

                for b in range(batch_obs.size(0)): # Iterate over sequences in the minibatch
                    seq_round_type = batch_initial_round_types[b]
                    seq_actions = batch_actions[b] # (sequence_length,)

                    if seq_round_type == "large":
                        seq_logits = new_logits_large_seq[b] # (sequence_length, action_dim_large)
                    elif seq_round_type == "small":
                        seq_logits = new_logits_small_seq[b] # (sequence_length, action_dim_small)
                    else:
                        raise ValueError(f"Unknown round_type in batch: {seq_round_type}")
                    
                    dist = Categorical(logits=seq_logits)
                    new_log_probs_list.append(dist.log_prob(seq_actions)) # (sequence_length,)
                    entropy_list.append(dist.entropy()) # (sequence_length,)
                
                # Stack to form tensors: (minibatch_size, sequence_length)
                new_log_probs = torch.stack(new_log_probs_list)
                entropies = torch.stack(entropy_list)
                mean_entropy = entropies.mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs) # batch_old_log_probs is also (minibatch, seq_len)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values_seq, batch_returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * mean_entropy

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        self.buffer.clear()

# --- Training Loop ---
def train():
    env = VaryingActionSpaceEnv(OBS_DIM_DUMMY, ACTION_DIM_LARGE, ACTION_DIM_SMALL)
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
content_copy
download
Use code with caution.
Python
Key Changes and Considerations:

VaryingActionSpaceEnv:
Alternates self.current_round_type (and thus self.action_space) upon reset().
Returns current_round_type in the info dictionary.
The reward function is slightly different for each round type to give the agent a reason to distinguish policies.
ActorCriticRNN:
Now has self.actor_head_large and self.actor_head_small.
forward() returns a tuple of logits: (logits_large, logits_small).
RolloutBufferRNN:
Added self.round_types to store the active round type for each step.
add() now takes round_type.
get_sequences() now yields batch_initial_round_types. This is an array containing the round type of the first step of each sequence in the batch.
Simplification/Assumption: We assume that the round type for the first step of a training sequence applies to the entire sequence for the purpose of selecting the correct policy head during the PPO update. If the round type could change within a SEQUENCE_LENGTH block and this was critical, the update logic would need to be more granular (processing each step of the sequence individually according to its specific round_type). For many practical scenarios where "rounds" are episode-like or phase-like, this simplification is often acceptable.
PPO_RNN_Agent:
select_action(): Takes current_round_type as input and uses it to select the correct logits from the policy output for sampling.
update(): This is the most significant change.
It receives batch_initial_round_types from the buffer.
It iterates through each sequence in the minibatch (for b in range(batch_obs.size(0))).
For each sequence, it uses batch_initial_round_types[b] to decide whether to use new_logits_large_seq[b] or new_logits_small_seq[b] for calculating new log probabilities and entropy for that entire sequence.
The lists new_log_probs_list and entropy_list are populated, then stacked into tensors for the PPO loss calculations.
Training Loop:
Retrieves current_round_type from env.reset() and env.step().
Passes current_round_type to agent.select_action() and agent.buffer.add().
This setup allows the agent to learn distinct behaviors for different action spaces while sharing the underlying LSTM representation and critic. The dummy environment is simple, but it demonstrates the mechanism. For a real environment, ensure the round_type signal is reliable and appropriately reflects the change in action space.
