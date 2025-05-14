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