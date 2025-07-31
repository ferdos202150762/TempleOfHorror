import sys
import json
import numpy as np
from tqdm import tqdm
from sys import argv

sys.path.append('GameModel')
from TempleOfHorror import TempleOfHorror

# Hyperparameters
if len(argv) < 3:
    print("Usage: python inference.py <number_of_episodes> <path_to_model.json>")
    sys.exit(1)

NUMBER_EPISODES = int(argv[1])
MODEL_PATH = argv[2]

def load_json_model(filepath):
    """Loads a JSON model from the specified file path."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {filepath}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        sys.exit(1)

def compute_hand_from_labels(hand):
    """Computes a sorted string representation of a player's hand."""
    type_map = {
        "gold": "Gold",
        "fire": "Fire",
        "empty": "Empty"
    }
    result = [type_map[item.split('_')[0].lower()] for item in hand if item.split('_')[0].lower() in type_map]
    priority = {"Gold": 0, "Fire": 1, "Empty": 2}
    result.sort(key=lambda x: priority[x])
    return "".join(result)

def get_history_key(env, player_id, history):
    """Generates the history key for the strategy lookup."""
    hand = compute_hand_from_labels(env.static_player_hands[f'agent_{player_id}'])
    role = env.player_role[f'agent_{player_id}'][:-2]
    return f"P:{player_id},R:{role},C:{hand}GameInits->(P:0{history}"

def get_ai_choice(env, player_id, history, strategies, is_message_phase):
    """Determines the AI's action based on the loaded strategy."""
    info_set_key = get_history_key(env, player_id, history)
    strategy = strategies.get(info_set_key)

    if is_message_phase:
        message_space = env.message_space
        if strategy and len(strategy) == len(message_space):
            choice_idx = np.random.choice(len(strategy), p=strategy)
            return message_space[choice_idx]
        return message_space[np.random.randint(len(message_space))]
    else: # Action phase
        action_space = env.action_spaces[f'agent_{player_id}']
        if strategy and len(strategy) == len(action_space):
            return np.random.choice(action_space, p=strategy)
        return np.random.choice(action_space)

# Load the trained model from JSON
strategies = load_json_model(MODEL_PATH)

count_wins_defender = 0
for episode in tqdm(range(NUMBER_EPISODES)):
    env = TempleOfHorror()
    state = env.reset()
    history = ""

    # Message exchange phase
    for agent_idx in range(env.N):
        message = get_ai_choice(env, agent_idx, history, strategies, is_message_phase=True)
        state = env.step_message(message)
        history += f",A:{message})->(P:{env.provide_message if env.provide_message != 0 else 0}"

    # Main game phase
    agent_key_idx = 0
    while not env.done:
        action = get_ai_choice(env, agent_key_idx, history, strategies, is_message_phase=False)
        done, next_state, reward, card, winner = env.step(action)
        history += f",A:{action},C:{card})->(P:{action}"
        state = next_state
        agent_key_idx = action

    if winner == 1:
        count_wins_defender += 1

print(f"Defender wins: {count_wins_defender / NUMBER_EPISODES:.2%}")
print(f"Attacker wins: {(NUMBER_EPISODES - count_wins_defender) / NUMBER_EPISODES:.2%}")
