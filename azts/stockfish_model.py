"""
fake inference giving values
according to stockfish engine
"""
import numpy as np

from Interpreter.game import evaluate_position, get_policy_from_position

# pylint: disable=W0613
# pylint: disable=R0201
# pylint: disable=R0903

# dimensions of mock object tensors
MOVE_DIMENSIONS = (8, 8, 64)
POS_DIMENSIONS = (8, 8, 11)

class StockfishModel():
    """
    fake inference giving values
    according to stockfish engine
    """

    def inference(self, position):
        '''

        :param np.array: current game position
        in tensor notation
        :return tuple: tuple containing np.array
        with policy tensor and int with position
        evaluation
        '''
        assert position.shape == POS_DIMENSIONS
        policy = get_policy_from_position(position)
        evaluation = evaluate_position(position)

        return (policy, evaluation)
# pylint: enable=W0613
# pylint: enable=R0201
# pylint: enable=R0903


if __name__ == "__main__":
    sf = StockfishModel()
    for i in range(10):
        pos = np.random.rand(*POS_DIMENSIONS)
        pol, ev = sf.inference(pos)
        print(pol, ev)