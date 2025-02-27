
from collections import Counter
from TDS_prompts import prompt_dict
from llm_utils import llm_call
import json
import time

import random
from typing import List, Dict


class TempleDesSchreckensGame:
    def __init__(self):
        """Initialize the game with num_players participants."""
        self.num_players = 3


        self.num_wachterinnen = 1

        roles = (["W"] * self.num_wachterinnen +
                 ["A"] * (self.num_players - self.num_wachterinnen))
        random.shuffle(roles)

        # Create a structure for players
        # We'll store each player's data in a dict with keys:
        # 'role', 'cards' (treasure cards each has), 'revealed_cards', etc.
        self.players = []

        for i in range(self.num_players):
            player_id = i + 1
            self.players.append({
                "id": player_id,
                "role": roles[i],  # 'W' or 'A'
                "cards": [],  # face-down treasure cards (strings)
                "revealed_cards": [],  # any revealed treasure cards so far
                "initial_claim": None  # 1, 0 for yes or no at least one trap
            })


        gold_cards = 4
        trap_cards = 2

        # Then fill the rest with empty:
        empty_cards = 0#total_cards_needed - gold_cards - trap_cards

        treasure_deck = (["G"] * gold_cards +
                         ["T"] * trap_cards )
        random.shuffle(treasure_deck)

        # Deal to players
        idx = 0
        for p in self.players:
            p["cards"] = treasure_deck[idx: idx + 2]
            idx += 2

        # Step 3: Set up trackers for game progress
        # ------------------------------------------------
        self.current_round = 1
        self.max_rounds = 4
        self.key_holder_index = random.choice([0,1,2])
        self.game_over = False

        # For counting globally revealed cards
        self.total_gold_revealed = 0
        self.total_traps_revealed = 0

        # Known total gold / traps from the deck
        self.total_gold_in_game = gold_cards
        self.total_traps_in_game = trap_cards

        self.initialize_game_state()

        # keep track of winner
        self.winner = ''

    def initialize_game_state(self):
        """
        Sets up the initial game_state with a public part (visible to all)
        and a private part for each player. The public part includes:
          - public_round_number
          - public_move_number_within_round
          - public_current_key_holder
          - public_player_claimed_traps (initially None, to be set/checked later)
          - public_player_traps_found
          - public_player_gold_found
          - public_player_has_lied
          - public_total_gold_found_so_far
          - public_total_traps_found_so_far

        The private part for each player i includes:
          - private_role
          - private_initial_cards
          - private_current_cards
        """
        self.game_state = {
            'public': {
                'round_number': self.current_round,
                'move_number_in_round': 0,
                'current_key_holder': self.players[self.key_holder_index]['id'],  # which player's ID holds the key
                'player_claimed_traps': [None] * self.num_players,  # Each player's initial trap claim, set later
                'player_traps_found': [0] * self.num_players,  # How many traps revealed from each player's area
                'player_gold_found': [0] * self.num_players,  # How many gold revealed from each player's area
                'player_has_lied': [False] * self.num_players,
                # Whether a player has been caught lying based on their claim vs. reveals
                'total_gold_found_so_far': 0,
                'total_traps_found_so_far': 0
            },
            'private': []
        }

        # Fill the private info for each of the three players
        for i, p in enumerate(self.players):
            private_data = {
                'private_role': p['role'],
                'private_player_number': i, # 'A' or 'W'
                'private_initial_cards': p['cards'][:],  # copy of the cards initially dealt
                'private_current_cards': p['cards'][:]  # copy of what they still hold
            }
            self.game_state['private'].append(private_data)

    def update_game_state(self):
        """
        Updates both the public and private parts of the game_state
        after each claim or turn:
          - public part: round number, move number in round, key holder, claims, found traps/gold, total found, lying flags
          - private part: player's current cards remain in sync with the engine
        """
        # Update the public round number and increment the move counter
        self.game_state['public']['round_number'] = self.current_round
        self.game_state['public']['move_number_in_round'] += 1

        # Update which player currently has the key (by ID)
        self.game_state['public']['current_key_holder'] = self.players[self.key_holder_index]['id']

        # Refresh each player's data
        for i, p in enumerate(self.players):

            # If not yet set, record their claimed trap status based on initial cards.
            # (In a more complex version, this might come from actual statements or bluffing.)
            if self.game_state['public']['player_claimed_traps'][i] is None and self.players[i]['initial_claim']:
                self.game_state['public']['player_claimed_traps'][i] = self.players[i]['initial_claim']

            # Count how many traps/gold have been revealed from that player's area
            traps_found = p['revealed_cards'].count('T')
            gold_found = p['revealed_cards'].count('G')
            self.game_state['public']['player_traps_found'][i] = traps_found
            self.game_state['public']['player_gold_found'][i] = gold_found

            # Check if the player's initial claim conflicts with what has been revealed
            claim = self.game_state['public']['player_claimed_traps'][i]
            # If claimed 'NoT' but at least one trap is revealed, that's a lie
            if claim == 'No trap' and traps_found > 0:
                self.game_state['public']['player_has_lied'][i] = True
            # If claimed 'T' but no trap has shown up in revealed cards (and at least something was revealed), also a lie
            if claim == 'At least 1 trap' and traps_found == 0 and len(p['revealed_cards']) > 1:
                self.game_state['public']['player_has_lied'][i] = True

            # Keep the private current_cards synced with what the engine says
            self.game_state['private'][i]['private_current_cards'] = p['cards'][:]

        # Update totals for gold and traps revealed so far
        self.game_state['public']['total_gold_found_so_far'] = self.total_gold_revealed
        self.game_state['public']['total_traps_found_so_far'] = self.total_traps_revealed

    def get_order_array(self,key_holder):
        if key_holder == 0:
            return [0,1,2]
        elif key_holder ==1:
            return [1,2,0]
        else:
            return [2,0,1]

    def play_game(self):
        """Run a full simulation of up to 4 rounds. Each round,
        every player gets exactly one turn with the key (random moves)."""
        print(f"Starting game with {self.num_players} players.")
        print(f"Wächterinnen: {self.num_wachterinnen}, Abenteurer: {self.num_players - self.num_wachterinnen}")
        print("Initial Roles (hidden to each other, but we’ll print for demonstration):")
        for p in self.players:
            print(f"  Player {p['id']-1} Role = {p['role']}  Cards = {p['cards']}")

        # reveal_action
        order_array = self.get_order_array(self.key_holder_index)
        for pl in order_array:
            self.reveal_action(pl)
            self.update_game_state()
            print(self.game_state)

        while not self.game_over and self.current_round <= self.max_rounds:
            print(f"\n===== ROUND {self.current_round} =====")

            # In each round, each player gets one turn as key holder
            for _ in range(self.num_players):
                if self.game_over:
                    break
                target_player = self.take_action()
                self.take_turn(target_player)
                self.update_game_state()
                print(self.game_state)

            self.current_round += 1

        # Determine game result if not ended prematurely
        if not self.game_over:
            # If we didn’t reveal all gold, Wächterinnen win
            if self.total_gold_revealed == self.total_gold_in_game:
                self.declare_winner("Abenteurer")
            else:
                self.declare_winner("Wächterinnen")

    def reveal_action(self,player):
        game_state = self.game_state['public']
        game_state.update(self.game_state['private'][player])
        game_state_json = json.dumps(game_state)
        prompt = prompt_dict["trap_claim_action_part1"].format(game_state=game_state_json) + prompt_dict["trap_claim_action_part2"]
        message  = [{'role':'system','content':prompt}]
        result = llm_call(message)
        json_result = json.loads(result)
        if player ==1:
            if json_result['claimed_trap']==1:
                self.players[player]["initial_claim"] = 'At least 1 trap'
            else:
                self.players[player]["initial_claim"] = 'No trap'
            pass
        else:
            print('reveal player ' + str(player))
            result = int(input())
            if result == 1:
                self.players[player]["initial_claim"] = 'At least 1 trap'
            else:
                self.players[player]["initial_claim"] = 'No trap'



    def take_action(self):
        """One turn for the current key holder, who picks a random card from another player."""
        key_holder = self.players[self.key_holder_index]

        print(f"\n  --> Player {key_holder['id']-1} (Role={key_holder['role']}) holds the key.")

        # Check if game already ended
        if self.game_over:
            return

        # Randomly pick a target who is NOT the key holder
        possible_targets = [p for p in self.players if p["id"] != key_holder["id"] and len(p["cards"]) > 0]
        if not possible_targets:
            # If no one has cards or no valid target, skip
            self.pass_key()
            return
        elif len(possible_targets)==1:
            player_index = possible_targets[0]["id"]-1
            return self.players[player_index]
        possible_targets_string ="["
        for item in possible_targets:
            possible_targets_string = possible_targets_string+ str(item["id"]-1)+","

        possible_targets_string = possible_targets_string[:-1]+"]"

        print(possible_targets_string)

        game_state = self.game_state['public']
        game_state.update(self.game_state['private'][self.key_holder_index])
        game_state_json = json.dumps(game_state)
        prompt = prompt_dict["next_player_action_part1"].format(game_state=game_state_json,targets = possible_targets_string) + prompt_dict[
            "next_player_action_part2"]
        message = [{'role': 'system', 'content': prompt}]
        result = llm_call(message)
        json_result = json.loads(result)
        result = json_result['next_player']
        if (key_holder['id']-1)==0 or (key_holder['id']-1)==2:
            print('chose target player for player  ' + str(key_holder['id']-1))
            result  = int(input())

        target_player = self.players[result]
        return target_player

    def take_turn(self, target_player = None):
        """One turn for the current key holder, who picks a random card from another player."""
        key_holder = self.players[self.key_holder_index]

        print(f"\n  --> Player {key_holder['id']-1} (Role={key_holder['role']}) holds the key.")

        # Check if game already ended
        if self.game_over:
            return
        # Randomly pick one face-down card from target
        if not target_player["cards"]:
            # If they have no cards, skip
            self.pass_key()
            return

        reveal_index = random.randrange(len(target_player["cards"]))
        revealed_card = target_player["cards"].pop(reveal_index)  # remove from face-down
        target_player["revealed_cards"].append(revealed_card)  # record it as revealed

        # Print info
        print(f"     Player {key_holder['id']-1} opens a room from Player {target_player['id']-1}:  -> {revealed_card}")

        # Update global trackers
        if revealed_card == "G":
            self.total_gold_revealed += 1
        elif revealed_card == "T":
            self.total_traps_revealed += 1

        # Check immediate end conditions
        if self.total_traps_revealed == self.total_traps_in_game:
            # All traps found => Wächterinnen win instantly
            print(f"     >>> All {self.total_traps_in_game} traps have been revealed!")
            self.declare_winner("Wächterinnen")
            return

        if self.total_gold_revealed == self.total_gold_in_game:
            # All gold found => Abenteurer win
            print(f"     >>> All {self.total_gold_in_game} gold have been found!")
            self.declare_winner("Abenteurer")
            return

        # Pass key to target player for next turn
        next_key_index = self.players.index(target_player)
        self.key_holder_index = next_key_index

    def pass_key(self):
        """If the key holder can’t open a valid card, just pass the key to the next seat."""
        next_index = (self.key_holder_index + 1) % self.num_players
        self.key_holder_index = next_index

    def declare_winner(self, side: str):
        """Declare the winner and end the game."""
        print(f"\nGAME OVER! The {side} win!")
        self.game_over = True
        self.winner = side
        # Reveal final details
        self.print_final_details()

    def print_final_details(self):
        """Show all players' final states (cards + revealed)."""
        print("\nFINAL PLAYER INFO:")
        for p in self.players:
            print(f" Player {p['id']-1} (Role={p['role']})")
            print(f"   Face-down Cards: {p['cards']}")
            print(f"   Revealed Cards: {p['revealed_cards']}")
        print(f"\nTotal Gold Revealed: {self.total_gold_revealed} / {self.total_gold_in_game}")
        print(f"Total Traps Revealed: {self.total_traps_revealed} / {self.total_traps_in_game}")
        print("-------------------------------------------------")

# -------------------------------------------------
# DEMO USAGE (uncomment if running as a standalone)
# -------------------------------------------------
if __name__ == "__main__":
    winners = []
    for i in range(1):
        print('<<<<<<<<<<>>>>>>>>>>')
        print('game: ' + str(i))
        seed = 777
        random.seed(seed)  # make it repeatable for debugging
        game = TempleDesSchreckensGame()
        game.play_game()

        winners.append(game.winner)
        time.sleep(2)
    print(winners)
    print(Counter(winners))

# -------------------------------------------------