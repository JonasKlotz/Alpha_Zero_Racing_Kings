"""
for testing alpha zero tree search
until we have real models
"""
import numpy as np
# pylint: disable=C0116
# pylint: disable=W0613
# pylint: disable=R0201
# pylint: disable=R0903

# dimensions of mock object tensors
DIMENSIONS = (8, 8, 64)

class MockModel():
    """
    for testing alpha zero tree search
    until we have real models
    """
    def inference(self, position):
        policy = np.random.rand(*DIMENSIONS)
        evaluation = np.random.rand(1)[0] - 0.5

        return (policy, evaluation)
# pylint: enable=C0116
# pylint: enable=W0613
# pylint: enable=R0201
# pylint: enable=R0903
