"""
fake inference giving values
according to stockfish engine
"""

from Interface.TensorNotation import tensor_to_fen, fen_to_tensor
from Interpreter import game

# pylint: disable=W0613
# pylint: disable=R0201
# pylint: disable=R0903

# dimensions of mock object tensors
MOVE_DIMENSIONS = (8, 8, 64)
POS_DIMENSIONS = (8, 8, 11)
# Provide relative path to Stockfish Engine
PATH_TO_ENGINE = "../Interpreter/Engine/stockfish-x86_64"


class StockfishModel():
    """
    fake inference giving values
    according to stockfish engine
    """

    def __init__(self, config):
        self.game = game.Game()
        self.time_limit = config.stockfish.time_limit
        self.search_depth = config.stockfish.search_depth

    def inference(self, position):
        """

        :param position: np.array: current game position
        in tensor notation
        :return tuple: tuple containing np.array
        with policy tensor and int with position
        evaluation
        """
        assert position.shape == POS_DIMENSIONS
        self.game.board.set_fen(tensor_to_fen(position))
        evaluation = self.game.get_evaluation(PATH_TO_ENGINE)
        policy = game.policy_to_tensor(game.normalize_policy(
            self.game.get_policy(
                PATH_TO_ENGINE, time_limit=self.time_limit, depth_limit=self.search_depth)))

        return (policy, evaluation)


# pylint: enable=W0613
# pylint: enable=R0201
# pylint: enable=R0903


if __name__ == "__main__":
    sf = StockfishModel()
    while not sf.game.is_ended():
        pol, ev = sf.inference(fen_to_tensor(sf.game.board.fen()))
        print(ev)
        sf.game.play_stockfish(0.01, PATH_TO_ENGINE)
    sf.game.show_game()
