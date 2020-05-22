import numpy as np

# dimensions of mock object tensors
DIMENSIONS = (8, 8, 64)

class MockModel():
    def inference(self, position):
        policy = np.random.rand(*DIMENSIONS)
        evaluation = np.random.rand(1)[0] - 0.5

        return (policy, evaluation)

