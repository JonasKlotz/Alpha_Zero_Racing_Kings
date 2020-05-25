"""
for testing alpha zero tree search
until we have real models
"""
import numpy as np
from Model.model import AZero as MockModel
# pylint: disable=C0116
# pylint: disable=W0613
# pylint: disable=R0201
# pylint: disable=R0903

# dimensions of mock object tensors
MOVE_DIMENSIONS = (8, 8, 64)
POS_DIMENSIONS = (8, 8, 11)

# class MockModel():
#     """
#     for testing alpha zero tree search
#     until we have real models
#     """
#     def inference(self, position):
#         assert position.shape == POS_DIMENSIONS
#         policy = np.random.rand(*MOVE_DIMENSIONS)
#         evaluation = np.random.rand(1)[0] - 0.5

#         return (policy, evaluation)
# pylint: enable=C0116
# pylint: enable=W0613
# pylint: enable=R0201
# pylint: enable=R0903
