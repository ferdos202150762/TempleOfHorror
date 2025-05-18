from open_spiel.python import games
from open_spiel.python.games import pyspiel

class CustomState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self._cur_player = 0
        self._is_terminal = False
        self._history = []
    
    def current_player(self):
        return self._cur_player
    
    def _legal_actions(self, player):
        return [0, 1]  # Example actions

    def _apply_action(self, action):
        self._history.append(action)
        if len(self._history) >= 5:
            self._is_terminal = True
    
    def _action_to_string(self, player, action):
        return f"Player {player} chooses action {action}"
    
    def is_terminal(self):
        return self._is_terminal
    
    def returns(self):
        return [1.0 if self._is_terminal else 0.0]

class CustomGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(pyspiel.GameType(
            short_name="custom_game",
            long_name="Custom Game",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
            information=pyspiel.GameType.Information.PERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.ZERO_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=2,
            min_num_players=2,
            provides_information_state_string=True
        ), params or {})
    
    def new_initial_state(self):
        return CustomState(self)

    def num_players(self):
        return 2

pyspiel.register_game("custom_game", CustomGame)
game = pyspiel.load_game("custom_game")