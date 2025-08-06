
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
MODELS_DIR = 'models/modelv2-defenders-could-lie'
NUM_SAMPLES = 100_000
attacker_truthful = True

# --- Evaluation Mode ---
# If True, the 'best responder' is a random player (measures exploitability).
# If False, the 'best responder' uses the model's strategy (self-play evaluation).
BEST_RESPONDER_IS_RANDOM = True

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
    For the "random" model, it evaluates a game with all random players.
    Returns the average utility of the random player and the 95% confidence interval for attackers and defenders.
    """
    is_random_benchmark = strategy_file == "random"
    model_name = "Random Player (Benchmark)" if is_random_benchmark else os.path.basename(strategy_file)
    print(f"\n--- Evaluating model: {model_name} ---")

    strategies = None
    if not is_random_benchmark:
        try:
            with open(strategy_file, 'r') as f:
                strategies = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading strategy file {strategy_file}: {e}")
            return None

    attacker_utilities = []
    defender_utilities = []

    for i in range(num_samples):
        best_responder_id = i % 3

        env = TempleOfHorror()
        state = env.reset()
        history = ""

        # --- Message Exchange Phase ---
        for agent_idx in range(env.N):
            is_best_responder = agent_idx == best_responder_id
            play_randomly = is_random_benchmark or (is_best_responder and BEST_RESPONDER_IS_RANDOM)

            if play_randomly:
                message = env.message_space[np.random.randint(len(env.message_space))]
            else:
                is_attacker = 'attacker' in env.player_role[f'agent_{agent_idx}']
                if attacker_truthful and is_attacker:

                    number_fire = env.enc_player_hands[f'agent_{agent_idx}'].count(2)
                    number_gold = env.enc_player_hands[f'agent_{agent_idx}'].count(3)

                    message = (number_fire,number_gold)


                else:
                    info_set_key = get_history_key(env, agent_idx, history)
                    strategy = strategies.get(info_set_key)
                    try:
                        message_idx = np.random.choice(len(env.message_space), p=strategy)
                    except:
                        print("Strategy not found",strategy)
                    
                    message = env.message_space[message_idx]


            
            state = env.step_message(message)
            history += f",A:{message})->(P:{env.provide_message if env.provide_message != 0 else 0}"

        # --- Main Game Phase ---
        agent_key_idx = 0
        while not env.done:
            agent_key = f"agent_{agent_key_idx}"
            available_actions = env.action_spaces[agent_key]

            is_best_responder = agent_key_idx == best_responder_id
            play_randomly = is_random_benchmark or (is_best_responder and BEST_RESPONDER_IS_RANDOM)

            if play_randomly:
                action = np.random.choice(available_actions)
            else:
                info_set_key = get_history_key(env, agent_key_idx, history)
                strategy = strategies.get(info_set_key)
                try:
                    action = np.random.choice(available_actions, p=strategy)
                except:
                    print("Strategy not found",history)

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
            utility = 100 if is_br_defender else -100
        else:  # Attackers win
            utility = -100 if is_br_defender else 100
        
        if is_br_defender:
            defender_utilities.append(utility)
        else:
            attacker_utilities.append(utility)

    results = {}
    if is_random_benchmark:
        player_type = "random"
        metric_name = "random-play utility"
    elif BEST_RESPONDER_IS_RANDOM:
        player_type = "random"
        metric_name = "exploitability"
    else: # not random benchmark and not random responder
        player_type = "model"
        metric_name = "self-play utility"

    if attacker_utilities:
        avg_attacker_utility = np.mean(attacker_utilities)
        ci_attacker = stats.t.interval(0.95, len(attacker_utilities)-1, loc=avg_attacker_utility, scale=stats.sem(attacker_utilities))
        print(f"Attacker - Average utility for {player_type} player ({metric_name}): {avg_attacker_utility:.4f}")
        print(f"Attacker - 95% confidence interval: ({ci_attacker[0]:.4f}, {ci_attacker[1]:.4f})")
        results['attacker'] = (avg_attacker_utility, ci_attacker)
    else:
        print("Attacker - No samples.")
        results['attacker'] = (None, None)

    if defender_utilities:
        avg_defender_utility = np.mean(defender_utilities)
        ci_defender = stats.t.interval(0.95, len(defender_utilities)-1, loc=avg_defender_utility, scale=stats.sem(defender_utilities))
        print(f"Defender - Average utility for {player_type} player ({metric_name}): {avg_defender_utility:.4f}")
        print(f"Defender - 95% confidence interval: ({ci_defender[0]:.4f}, {ci_defender[1]:.4f})")
        results['defender'] = (avg_defender_utility, ci_defender)
    else:
        print("Defender - No samples.")
        results['defender'] = (None, None)
    
    return results

def main():
    """
    Main function to find and evaluate all models.
    """
    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, '*.json')))
    
    if not model_files:
        print(f"No model files found in '{MODELS_DIR}'")

    print(f"Found {len(model_files)} models to evaluate in '{MODELS_DIR}'.")
    
    all_results = {}

    # Add evaluation for the random player benchmark
    random_results = evaluate_model("random", NUM_SAMPLES)
    if random_results:
        all_results["Random Player (Benchmark)"] = random_results

    for model_file in model_files:
        eval_results = evaluate_model(model_file, NUM_SAMPLES)
        if eval_results:
            all_results[os.path.basename(model_file)] = eval_results
            
    print("\n\n" + "="*30)
    print("  Evaluation Summary")
    print("="*30)
    if not all_results:
        print("No models were successfully evaluated.")
    else:
        ci_str = "95% CI"
        for model, results in all_results.items():
            print(f"  - {model}:")
            if "Benchmark" in model:
                metric_name = "Random-Play Utility"
            elif BEST_RESPONDER_IS_RANDOM:
                 metric_name = "Exploitability"
            else:
                metric_name = "Self-Play Utility"

            if results['attacker'][0] is not None:
                value, confidence_interval = results['attacker']
                print(f"    Attacker {metric_name}: {value:.4f} ({ci_str}: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}])")
            if results['defender'][0] is not None:
                value, confidence_interval = results['defender']
                print(f"    Defender {metric_name}: {value:.4f} ({ci_str}: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}])")
    print("="*30)
    if BEST_RESPONDER_IS_RANDOM:
        print("\nExploitability is measured as the average utility a random player achieves against a group.")
        print("A higher value suggests the model is more exploitable.")
    else:
        print("\nSelf-play utility is measured as the average utility a model player achieves against other model players.")


if __name__ == "__main__":
    main()

