import streamlit as st
import sys
import json
import numpy as np
import time
import random
import os

# Ensure the GameModel directory is in the Python path
sys.path.append('GameModel')
from TempleOfHorror import TempleOfHorror

# --- Configuration ---
MODELS_DIR = 'CFR/models/modelv2-defenders-could-lie/'

# --- Helper Functions ---

def get_model_files():
    """Returns a list of available model files."""
    return [f for f in os.listdir(MODELS_DIR) if f.endswith('.json')]

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

def get_ai_choice(env, player_id, history, strategies):
    """Determines the AI's action based on the loaded strategy."""
    info_set_key = get_history_key(env, player_id, history)
    strategy = strategies.get(info_set_key)

    if env.message_provided: # Action phase
        action_space = env.action_spaces[f'agent_{player_id}']
        if strategy and len(strategy) == len(action_space):
            return np.random.choice(action_space, p=strategy)
        return np.random.choice(action_space)
    else: # Message phase
        message_space = env.message_space
        role = env.player_role[f'agent_{player_id}']
        if role.startswith('attacker'):
            hand = env.player_hands[f'agent_{player_id}']
            num_fire = sum(1 for card in hand if card.startswith('fire'))
            num_gold = sum(1 for card in hand if card.startswith('gold'))
            message_space = [(num_fire, num_gold)]

        if strategy and len(strategy) == len(message_space):
            choice_idx = np.random.choice(len(strategy), p=strategy)
            return message_space[choice_idx]
        return message_space[np.random.randint(len(message_space))]


# --- UI Rendering Functions ---

def render_sidebar(env):
    """Renders the sidebar with player information and score."""
    with st.sidebar:
        st.header(f"Your Information (Player {st.session_state.human_player_id})")
        role = env.player_role[f'agent_{st.session_state.human_player_id}'].split('_')[0].capitalize()
        st.write(f"**Your Role:** {role}")

        remaining_cards = [card.split('_')[0].capitalize() for card in env.player_hands[f'agent_{st.session_state.human_player_id}']]
        st.write(f"**Your Hand:** {', '.join(remaining_cards) if remaining_cards else 'No cards left'}")

        st.header("Current Score")
        st.progress(env.score['gold'] / 3, text=f"Gold 💰 Cards: {env.score['gold']} / 3")
        st.progress(env.score['fire'] / 2, text=f"Fire 🔥 Cards: {env.score['fire']} / 2")

        if st.session_state.get('game_over'):
            st.header("Winner")
            winner_role = "Defender" if st.session_state.winner == 1 else "Attackers"
            st.success(f"{winner_role} win!")

def render_game_state(env, revealed_messages, action_history):
    """Renders the current state of the game."""
    st.header("Table Information")
    for i in range(env.N):
        player_label = f"Player {i}" + (" (You)" if i == st.session_state.human_player_id else "")
        opened_cards = env.public_observation[f'agent_{i}']
        st.write(f"**{player_label}:**")
        st.write(f"  - Empty: {opened_cards[0]}, Fire 🔥: {opened_cards[1]}, Gold 💰: {opened_cards[2]}")

    if revealed_messages or action_history:
        st.header("History of Play")
        if revealed_messages:
            st.write("**Revealed Messages:**")
            for i, msg in enumerate(revealed_messages):
                player_label = f"Player {i}" + (" (You)" if i == st.session_state.human_player_id else "")
                st.write(f"- {player_label} declared (Fire 🔥, Gold 💰): `{msg}`")
        if action_history:
            st.write("**Actions Taken:**")
            for i, (actor, target, card) in enumerate(action_history):
                actor_label = f"Player {actor}" + (" (You)" if actor == st.session_state.human_player_id else "")
                target_label = f"Player {target}" + (" (You)" if target == st.session_state.human_player_id else "")
                card_emoji = "🔥" if card.lower() == "fire" else "💰" if card.lower() == "gold" else ""
                st.write(f"- Turn {i+1}: {actor_label} chose {target_label}, revealing **{card.capitalize()} {card_emoji}**.")

# --- Main Application Logic ---

def main():
    st.title("⚔️ Temple of Horror ⚔️")

    if 'human_player_id' not in st.session_state:
        model_files = get_model_files()
        if not model_files:
            st.error("No model files found in the 'CFR/models' directory.")
            return

        st.session_state.selected_model = st.selectbox("Choose a model to play against:", model_files)

        if st.button("Start Game"):
            st.session_state.human_player_id = random.randint(0, 2)
            st.rerun()
        return

    # --- Game Initialization ---
    if 'game' not in st.session_state:
        try:
            strategy_file = os.path.join(MODELS_DIR, st.session_state.selected_model)
            with open(strategy_file, 'r') as f:
                st.session_state.strategies = json.load(f)
        except FileNotFoundError:
            st.error(f"Strategy file not found at {strategy_file}. Please ensure the file is in the correct location.")
            return

        game = TempleOfHorror()
        game.reset()
        st.session_state.game = game
        st.session_state.history = ""
        st.session_state.revealed_messages = []
        st.session_state.action_history = []
        st.session_state.current_player = 0
        st.session_state.game_over = False
        st.session_state.winner = None

    # Load from session state for easier access
    game = st.session_state.game
    strategies = st.session_state.strategies

    render_sidebar(game)

    # --- Game Over Screen ---
    if st.session_state.game_over:
        st.header("🎉 Game Over! 🎉")
        winner_role = "Defender" if st.session_state.winner == 1 else "Attackers"
        st.subheader(f"The {winner_role} Win!")

        st.write("**Final Roles & Hands:**")
        for i in range(game.N):
            role = game.player_role[f'agent_{i}'].split('_')[0].capitalize()
            player_label = f"Player {i}" + (" (You)" if i == st.session_state.human_player_id else "")
            
            # Get the original hand from static_player_hands
            original_hand = []
            for card in game.static_player_hands[f'agent_{i}']:
                card_name = card.split('_')[0].capitalize()
                if card_name == "Fire":
                    original_hand.append(f"{card_name} 🔥")
                elif card_name == "Gold":
                    original_hand.append(f"{card_name} 💰")
                else:
                    original_hand.append(card_name)
            hand_str = ", ".join(original_hand)
            
            st.write(f"- {player_label}: {role} - Hand: {hand_str}")

        st.markdown("---")
        render_game_state(game, st.session_state.revealed_messages, st.session_state.action_history)

        if st.button("Play Again"):
            for key in ['game', 'history', 'revealed_messages', 'action_history', 'current_player', 'game_over', 'winner', 'human_player_id']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return

    # --- Main Game UI ---
    render_game_state(game, st.session_state.revealed_messages, st.session_state.action_history)
    st.markdown("---")

    current_player = st.session_state.current_player
    player_label = f"Player {current_player}" + (" (You)" if current_player == st.session_state.human_player_id else "")

    # --- Phase Control ---
    if not game.message_provided:
        # --- MESSAGE PHASE ---
        st.header(f"Message Phase: {player_label}'s Turn")

        if current_player == st.session_state.human_player_id:
            st.write("Declare the number of fire 🔥 and gold 💰 cards you claim to have.")
            msg_options = {f"{msg[0]} Fire{' ' + '🔥' * msg[0] if msg[0] > 0 else ''}, {msg[1]} Gold{' ' + '💰' * msg[1] if msg[1] > 0 else ''}": tuple(msg) for msg in game.message_space}
            chosen_msg_str = st.radio("Choose your message:", options=msg_options.keys(), horizontal=True)

            if st.button("Send Message", type="primary"):
                message = msg_options[chosen_msg_str]
                st.session_state.revealed_messages.append(message)
                game.step_message(message)
                st.session_state.history += f",A:{message})->(P:{game.provide_message if game.provide_message != 0 else 0}"
                if game.message_provided:
                    st.session_state.current_player = 0
                else:
                    st.session_state.current_player += 1
                st.rerun()
        else: # AI's turn
            st.write(f"Waiting for Player {current_player} (AI) to send a message...")
            time.sleep(5)
            message = get_ai_choice(game, current_player, st.session_state.history, strategies)
            st.session_state.revealed_messages.append(message)
            game.step_message(message)
            st.session_state.history += f",A:{message})->(P:{game.provide_message if game.provide_message != 0 else 0}"
            if game.message_provided:
                st.session_state.current_player = 0 # Reset for action phase
            else:
                st.session_state.current_player += 1
            st.rerun()

    else:
        # --- ACTION PHASE ---
        st.header(f"Action Phase: {player_label}'s Turn")

        if current_player == st.session_state.human_player_id:
            st.write("Choose a player to reveal one of their cards.")
            action_space = game.action_spaces[f'agent_{current_player}']
            
            if not action_space:
                st.warning("You have no available actions. This shouldn't happen in a normal game.")
                return

            action = st.radio("Select a player:", options=action_space, format_func=lambda x: f"Player {x}", horizontal=True)

            if st.button("Reveal Card", type="primary"):
                done, _, _, card, winner = game.step(action)
                st.session_state.action_history.append((current_player, action, card))
                st.session_state.history += f",A:{action},C:{card})->(P:{action}"
                st.session_state.current_player = action # Next player is the one chosen

                if done:
                    st.session_state.game_over = True
                    st.session_state.winner = winner
                st.rerun()

        else: # AI's turn
            st.write(f"Waiting for Player {current_player} (AI) to act...")
            time.sleep(5)
            action = get_ai_choice(game, current_player, st.session_state.history, strategies)
            done, _, _, card, winner = game.step(action)
            st.session_state.action_history.append((current_player, action, card))
            st.session_state.history += f",A:{action},C:{card})->(P:{action}"
            st.session_state.current_player = action

            if done:
                st.session_state.game_over = True
                st.session_state.winner = winner
            st.rerun()

if __name__ == "__main__":
    main()
