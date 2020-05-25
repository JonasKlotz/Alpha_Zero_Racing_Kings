from azts import azts_tree
from azts import state_machine
from azts import mock_model

from azts.config import RUNS_PER_MOVE, WHITE


class Player():
    def __init__(self, color, model, runs_per_move=RUNS_PER_MOVE):
        self.statemachine = state_machine.StateMachine()
        self.model = model
        self.tree = azts_tree.AztsTree(self.statemachine,
                                       self.model,
                                       color,
                                       runs_per_move)

    def make_move(self):
        return self.tree.make_move()

    def receive_move(self, move):
        return self.tree.receive_move(move)

    def game_over(self):
        return self.tree.game_over()

    def game_state(self):
        return self.tree.game_state()

    def set_game_state(self, fen_position):
        self.tree.set_to_fen_state(fen_position)

    def dump_data(self):
        return [self.tree.get_position(),
                self.tree.get_policy_tensor(),
                None]


if __name__ == "__main__":
    player = Player(WHITE, mock_model.MockModel)
    print(f"First move of white player is {player.make_move()}.")
