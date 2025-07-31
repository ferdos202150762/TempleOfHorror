import sys
sys.path.append('GameModel')
from TempleOfHorror import TempleOfHorror
import numpy as np
import json

# --- Configuration ---
STRATEGY_FILE = 'CFR/models/cfr_strategies_10_1000.json'
HUMAN_PLAYER_ID = 0

# --- Helper Functions ---

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

def print_game_state(env, revealed_messages, action_history):
    """Prints the current state of the game for the human player."""
    print("\n" + "=" * 30)
    print("          GAME STATE          ")
    print("=" * 30)

    # Your Info
    print(f"\n--- Your Information (Player {HUMAN_PLAYER_ID}) ---")
    print(f"Your Role: {env.player_role[f'agent_{HUMAN_PLAYER_ID}'].split('_')[0].capitalize()}")

    # Display remaining cards in hand
    remaining_cards = [card.split('_')[0].capitalize() for card in env.player_hands[f'agent_{HUMAN_PLAYER_ID}']]
    print(f"Your Hand: {', '.join(remaining_cards) if remaining_cards else 'No cards left'}")

    # Table Info
    print("\n--- Table Information ---")
    for i in range(env.N):
        player_label = f"Player {i}" + (" (You)" if i == HUMAN_PLAYER_ID else "")
        opened_cards = env.public_observation[f'agent_{i}']
        print(f"{player_label}:")
        print(f"  - Empty: {opened_cards[0]}")
        print(f"  - Fire:  {opened_cards[1]}")
        print(f"  - Gold:  {opened_cards[2]}")

    # Score
    print("\n--- Current Score ---")
    print(f"Total Gold Cards Found: {env.score['gold']} / 3")
    print(f"Total Fire Cards Found: {env.score['fire']} / 2")

    # History of Play
    if revealed_messages:
        print("\n--- History of Play ---")
        print("Revealed Messages:")
        for i, msg in enumerate(revealed_messages):
            player_label = f"Player {i}" + (" (You)" if i == HUMAN_PLAYER_ID else "")
            print(f"  {player_label} declared (Fire, Gold): {msg}")

    if action_history:
        if not revealed_messages:
             print("\n--- History of Play ---")
        print("Actions Taken:")
        for i, (actor, target, card) in enumerate(action_history):
            actor_label = f"Player {actor}" + (" (You)" if actor == HUMAN_PLAYER_ID else "")
            target_label = f"Player {target}" + (" (You)" if target == HUMAN_PLAYER_ID else "")
            print(f"  Turn {i+1}: {actor_label} chose {target_label}, revealing '{card.capitalize()}'.")

    print("=" * 30 + "\n")

# --- Main Game Logic ---

def main():
    """Main function to run the interactive game."""
    # Load the strategy
    try:
        with open(STRATEGY_FILE, 'r') as f:
            strategies = json.load(f)
    except FileNotFoundError:
        print(f"Error: Strategy file not found at {STRATEGY_FILE}")
        return

    # Initialize environment
    env = TempleOfHorror()
    state = env.reset()
    history = ""
    revealed_messages = []
    action_history = []

    print("=" * 50)
    print("      WELCOME TO THE TEMPLE OF HORROR      ")
    print("=" * 50)
    print(f"You are Player {HUMAN_PLAYER_ID}. You will play against 2 AI opponents.")
    print("The game begins with the message exchange phase.")

    # --- Message Exchange Phase ---
    for agent_idx in range(env.N):
        agent_key = f"agent_{agent_idx}"

        if agent_idx == HUMAN_PLAYER_ID:
            print_game_state(env, revealed_messages, action_history)
            print(f"--- Your Turn to Send a Message (Player {HUMAN_PLAYER_ID}) ---")
            print("You need to declare a number of fire and gold cards.")
            print("Available messages (fire, gold):")
            for i, msg in enumerate(env.message_space):
                print(f"  {i}: {msg}")

            while True:
                try:
                    choice = int(input(f"Choose a message index [0-{len(env.message_space) - 1}]: "))
                    if 0 <= choice < len(env.message_space):
                        message = env.message_space[choice]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            print(f"You chose message: {message}")

        else: # AI Player's turn
            info_set_key = get_history_key(env, agent_idx, history)
            strategy = strategies.get(info_set_key)

            if strategy:
                message_idx = np.random.choice(len(strategy), p=strategy)
                message = env.message_space[message_idx]
            else:
                # Fallback to a random message if state not in strategy
                message = env.message_space[np.random.randint(len(env.message_space))]
            print(f"Player {agent_idx} chose message: {message}")

        revealed_messages.append(message)
        state = env.step_message(message)
        history += f",A:{message})->(P:{env.provide_message if env.provide_message != 0 else 0}"

    print("\n" + "=" * 50)
    print("      MESSAGE PHASE COMPLETE. MAIN GAME BEGINS.      ")
    print("=" * 50)

    # --- Main Game Phase ---
    agent_key_idx = 0
    while not env.done:
        agent_key = f"agent_{agent_key_idx}"

        if agent_key_idx == HUMAN_PLAYER_ID:
            print_game_state(env, revealed_messages, action_history)
            print(f"--- Your Turn to Act (Player {HUMAN_PLAYER_ID}) ---")
            print("Choose a player to reveal one of their cards.")

            available_actions = env.action_spaces[agent_key]
            print(f"Available players to choose from: {available_actions}")

            while True:
                try:
                    action = int(input(f"Enter player number {available_actions}: "))
                    if action in available_actions:
                        break
                    else:
                        print("Invalid choice. That player cannot be chosen.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            print(f"You chose to reveal a card from Player {action}.")

        else: # AI Player's turn
            print_game_state(env, revealed_messages, action_history)
            print(f"--- Player {agent_key_idx}'s Turn to Act ---")
            info_set_key = get_history_key(env, agent_key_idx, history)
            strategy = strategies.get(info_set_key)

            if strategy:
                action = np.random.choice(env.action_spaces[agent_key], p=strategy)
            else:
                # Fallback to a random action
                action = np.random.choice(env.action_spaces[agent_key])
            print(f"Player {agent_key_idx} chose to reveal a card from Player {action}.")

        done, next_state, reward, card, winner = env.step(action)
        print(f"A '{card.capitalize()}' card was revealed from Player {action}!")
        action_history.append((agent_key_idx, action, card))

        history += f",A:{action},C:{card})->(P:{action}"
        state = next_state
        agent_key_idx = action

    # --- End of Game ---
    print("\n" + "=" * 50)
    print("               GAME OVER               ")
    print("=" * 50)
    print_game_state(env, revealed_messages, action_history)

    print("\n--- Final Roles ---")
    for i in range(env.N):
        role = env.player_role[f'agent_{i}'].split('_')[0].capitalize()
        player_label = f"Player {i}" + (" (You)" if i == HUMAN_PLAYER_ID else "")
        print(f"  {player_label}: {role}")
    print()

    if winner == 1: # Defender wins
        print("🎉 DEFENDER WINS! 🎉")
        if env.player_role[f'agent_{HUMAN_PLAYER_ID}'] == 'defender_1':
            print("Congratulations, you have protected the temple!")
        else:
            print("The attackers have failed to secure the treasure.")
    else: # Attacker wins
        print("🎉 ATTACKERS WIN! 🎉")
        if 'attacker' in env.player_role[f'agent_{HUMAN_PLAYER_ID}']:
            print("Congratulations, you have found the treasure!")
        else:
            print("The defender has failed to protect the temple.")

if __name__ == "__main__":
    main()