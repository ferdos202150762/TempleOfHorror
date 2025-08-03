"""
This script evaluates the performance of CFR models for the Temple of Horror game.

It calculates the exploitability of each model by simulating games where a 
"best response" player plays against two players using the model's strategy.
The "best response" is approximated by a player who chooses actions randomly.

The script iterates through all model files in a specified directory,
runs a number of simulations for each, and reports the average utility 
achieved by the random player. This average utility serves as a measure of
the model's exploitability.

Usage:
    python evaluate_models.py
"""
import sys
sys.path.append("../GameModel")
import os
import json
import numpy as np
import glob
from scipy import stats

# --- Path Setup ---
# Add project directories to Python path to allow importing game model.
# This is based on the project structure and might need adjustment.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append('GameModel') # As seen in best_response_game.py

try:
    from TempleOfHorror import TempleOfHorror
except ImportError:
    print("Error: Could not import 'TempleOfHorror'.")
    print("Please ensure that the 'TempleOfHorror.py' file or 'GameModel' directory is accessible in the Python path.")
    sys.exit(1)

# --- Configuration ---
MODELS_DIR = 'models/modelv1'
NUM_SAMPLES = 2000

# --- Helper Functions (from best_response_game.py) ---

def compute_hand_from_labels(hand):
    """Computes a sorted string representation of a player's hand."""
    type_map = {"gold": "Gold", "fire": "Fire", "empty": "Empty"}
    result = [type_map[item.split('_')[0].lower()] for item in hand if item.split('_')[0].lower() in type_map]
    priority = {"Gold": 0, "Fire": 1, "Empty": 2}
    result.sort(key=lambda x: priority[x])
    return "".join(result)

def get_history_key(env, player_id, history):
    """Generates the history key for the strategy lookup."""
    hand = compute_hand_from_labels(env.static_player_hands[f'agent_{player_id}'])
    role = env.player_role[f'agent_{player_id}'][: -2]
    return f"P:{player_id},R:{role},C:{hand}GameInits->(P:0{history}"

# --- Main Evaluation Logic ---

def evaluate_model(strategy_file, num_samples):
    """
    Evaluates a model's exploitability by pitting it against a random player.
    Returns the average utility of the random player and the 95% confidence interval.
    """
    print(f"\n--- Evaluating model: {os.path.basename(strategy_file)} ---")

    try:
        with open(strategy_file, 'r') as f:
            strategies = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading strategy file {strategy_file}: {e}")
        return None, None

    utilities = []

    for i in range(num_samples):
        best_responder_id = i % 3

        env = TempleOfHorror()
        state = env.reset()
        history = ""

        # --- Message Exchange Phase ---
        for agent_idx in range(env.N):
            if agent_idx == best_responder_id:
                message = env.message_space[np.random.randint(len(env.message_space))]
            else:
                info_set_key = get_history_key(env, agent_idx, history)
                strategy = strategies.get(info_set_key)
                if strategy and len(strategy) == len(env.message_space):
                    message_idx = np.random.choice(len(env.message_space), p=strategy)
                    message = env.message_space[message_idx]
                else:
                    message = env.message_space[np.random.randint(len(env.message_space))]
            
            state = env.step_message(message)
            history += f",A:{message})->(P:{env.provide_message if env.provide_message != 0 else 0}"

        # --- Main Game Phase ---
        agent_key_idx = 0
        while not env.done:
            agent_key = f"agent_{agent_key_idx}"
            available_actions = env.action_spaces[agent_key]

            if agent_key_idx == best_responder_id:
                action = np.random.choice(available_actions)
            else:
                info_set_key = get_history_key(env, agent_key_idx, history)
                strategy = strategies.get(info_set_key)
                if strategy and len(strategy) == len(available_actions):
                    action = np.random.choice(available_actions, p=strategy)
                else:
                    action = np.random.choice(available_actions)

            done, _, _, card, winner = env.step(action)
            history += f",A:{action},C:{card})->(P:{action}"
            agent_key_idx = action

        # --- End of Game: Calculate utility for the best responder ---
        defender_id = -1
        for p_idx in range(env.N):
            if 'defender' in env.player_role[f'agent_{p_idx}']:
                defender_id = p_idx
                break
        
        is_br_defender = (best_responder_id == defender_id)
        
        if winner == 1:  # Defender wins
            utility = 1 if is_br_defender else -1
        else:  # Attackers win
            utility = -1 if is_br_defender else 1
        utilities.append(utility)

    avg_exploitability = np.mean(utilities)
    confidence_interval = stats.t.interval(0.95, len(utilities)-1, loc=avg_exploitability, scale=stats.sem(utilities))
    
    print(f"Average utility for random player (exploitability): {avg_exploitability:.4f}")
    print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
    
    return avg_exploitability, confidence_interval

def main():
    """
    Main function to find and evaluate all models.
    """
    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, '*.json')))
    
    if not model_files:
        print(f"No model files found in '{MODELS_DIR}'")
        return

    print(f"Found {len(model_files)} models to evaluate in '{MODELS_DIR}'.")
    
    results = {}
    for model_file in model_files:
        avg_utility, confidence_interval = evaluate_model(model_file, NUM_SAMPLES)
        if avg_utility is not None:
            results[os.path.basename(model_file)] = (avg_utility, confidence_interval)
            
    print("\n\n" + "="*30)
    print("  Evaluation Summary")
    print("="*30)
    if not results:
        print("No models were successfully evaluated.")
    else:
        for model, (exploitability, confidence_interval) in results.items():
            print(f"  - {model}: {exploitability:.4f} (95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}])")
    print("="*30)
    print("\nExploitability is measured as the average utility a random player achieves.")
    print("A higher value suggests the model is more exploitable.")


if __name__ == "__main__":
    main()
