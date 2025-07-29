import torch
# --- Hyperparameters ---
# ENV_NAME (will use dummy)
ACTION_DIM_LARGE = 6
ACTION_DIM_SMALL = 2
OBS_DIM_DUMMY = 4
HIDDEN_SIZE = 64
LR = 3e-4
GAMMA = 0.98
GAE_LAMBDA = 0.99
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
ROLLOUT_STEPS = 128
SEQUENCE_LENGTH = 10
ENTROPY_COEF = 0.05
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

TOTAL_TIMESTEPS = 1_000 # Adjusted for potentially slower learning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Inside your PPO_RNN_Agent class

def save_model(self, filepath_prefix):
    """Saves the policy network and optimizer state."""
    torch.save(self.policy.state_dict(), f"{filepath_prefix}_policy.pth")
    torch.save(self.optimizer.state_dict(), f"{filepath_prefix}_optimizer.pth")
    print(f"Model saved to {filepath_prefix}_policy.pth and {filepath_prefix}_optimizer.pth")

def load_model(self, filepath_prefix):
    """Loads the policy network and optimizer state."""
    try:
        self.policy.load_state_dict(torch.load(f"{filepath_prefix}_policy.pth", map_location=DEVICE))
        self.optimizer.load_state_dict(torch.load(f"{filepath_prefix}_optimizer.pth", map_location=DEVICE))
        self.policy.to(DEVICE) # Ensure model is on the correct device
        # If optimizer has tensors, they need to be moved to device too if not handled by load_state_dict
        # For Adam, this is usually handled, but good to be aware.
        print(f"Model loaded from {filepath_prefix}_policy.pth and {filepath_prefix}_optimizer.pth")
    except FileNotFoundError:
        print(f"Error: Model files not found at {filepath_prefix}. Starting from scratch.")
    except Exception as e:
        print(f"Error loading model: {e}. Starting from scratch.")

# You might also want to save training progress like total_timesteps, episode_num
# if you plan to resume training. This can be done by pickling these variables
# alongside the model weights.