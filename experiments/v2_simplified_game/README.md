# Temple of Horror - Experiment v2

This repository contains an experiment for the game "Temple of Horror". This version of the game has been simplified for experimental purposes.

## Game Description

"Temple of Horror" is a cooperative game with a hidden traitor element. The game is played with 3 players, where one player is randomly assigned the role of "defender" and the other two are "attackers".

### Game Components

*   **Deck of Cards:** The game uses a deck of 9 cards with the following distribution:
    *   4 Empty Cards
    *   3 Gold Cards
    *   2 Fire Cards
*   **Player Roles:**
    *   **Attacker:** Wins if 3 gold cards are revealed.
    *   **Defender:** Wins if 2 fire cards are revealed.

### Gameplay

1.  **Card Distribution:** At the beginning of the game, the 9 cards are shuffled and distributed evenly among the 3 players, so each player has a hand of 3 cards.
2.  **Messaging Phase:** Before players start taking actions, there is a messaging phase where each player can announce the number of gold and fire cards they have in their hand. This information is public to all players.
3.  **Action Phase:** Players take turns choosing another player to reveal a card from their hand. The chosen player randomly reveals one of their cards.
4.  **Winning Conditions:**
    *   The **attackers** win if a total of 3 **gold cards** are revealed.
    *   The **defender** wins if a total of 2 **fire cards** are revealed.
    *   The **defender** also wins if all 4 **empty cards**, 1 **fire card** and 2 **gold cards** are revealed.

## Experiment Goal

The main goal of this experiment is to analyze and model the strategic decisions of the players in this simplified version of "Temple of Horror". The experiment explores different approaches to modeling player behavior, including:

*   **CFR (Counterfactual Regret Minimization):** This approach uses the CFR algorithm to find the optimal strategies for each player.
*   **MARL (Multi-Agent Reinforcement Learning):** This approach uses multi-agent reinforcement learning to train agents to play the game.

## Project Structure

The repository is organized into the following directories:

*   `CFR/`: Contains the implementation of the CFR algorithm for the "Temple of Horror" game.
    *   `CFRalgorithm.py`: The core implementation of the CFR algorithm.
    *   `training.py`: The script for training the CFR model.
    *   `cfr_strategies.json`: The learned strategies from the CFR algorithm.
*   `GameModel/`: Contains the implementation of the "Temple of Horror" game itself.
    *   `TempleOfHorror.py`: The main file defining the game logic, rules, and environment.
*   `MARLv1/`: Contains the implementation of the MARL-based approach.
    *   `PPO_attacker.py`: The implementation of the PPO algorithm for the attacker agents.
    *   `PPO_defender.py`: The implementation of the PPO algorithm for the defender agent.
    *   `training.py`: The script for training the MARL agents.
*   `inference.py`: A script for running inference with the trained models and simulating games.

## How to Run

To run the experiment, you can use the following scripts:

*   **CFR Training:** `python CFR/training.py`
*   **MARL Training:** `python MARLv1/training.py`
*   **Inference:** `python inference.py`
