# Temple of Horror - Game Theory and AI Agents

This repository explores the game "Temple of Horror" through the lens of game theory and artificial intelligence. It includes implementations of the game, various AI agents to play it, and experiments with different game versions.

## Project Structure

The repository is organized as follows:

*   **/cfr_solver**: Contains the implementation of a Counterfactual Regret Minimization (CFR) solver for the game. CFR is an algorithm used to find a Nash Equilibrium in two-player zero-sum games.
*   **/game_model**: Contains the core Python implementation of the "Temple of Horror" game logic.
*   **/experiments**: Contains different versions of the game and AI agent implementations.
    *   **/v0_original_game**: The original version of the game.
    *   **/v1_simplified_game**: A simplified version of the game, used for initial AI agent development.
    *   **/v2_simplified_game**: A further simplified version of the game, with a focus on balancing and AI agent training.
    *   **/v3_simplified_game**: The latest simplified version of the game, used for developing a CFR-based agent.

## Getting Started

To get started with this project, you will need to have Python installed. It is recommended to use a virtual environment to manage dependencies.

## Running the Game

To play the game, you can run the `TempleOfHorror.py` script in the `game_model` directory:

```bash
python game_model/TempleOfHorror.py
```

## AI Agents

This repository includes several AI agents to play the game:

*   **CFR Agent**: A game theory-based agent that uses the Counterfactual Regret Minimization algorithm to play optimally.
*   **MARL Agents**: Agents based on Multi-Agent Reinforcement Learning (MARL), which learn to play the game by interacting with each other.

## Experiments

The `experiments` directory contains different versions of the game and AI agent implementations. Each version has its own set of files and may require different dependencies. Please refer to the README files within each experiment's directory for more information.

## Contributing

Contributions to this project are welcome. Please feel free to open an issue or submit a pull request.
