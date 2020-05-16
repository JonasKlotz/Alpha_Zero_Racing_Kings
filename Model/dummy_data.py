import numpy as np
import pickle

DIM_POS = (8, 8, 11)
DIM_MOVE = (8, 8, 64)


def position():
    position = np.random.rand(*DIM_POS)
    position[position > 0.8] = 1
    position[position < 0.9] = 0
    position = position.astype(np.uint8)
    return position

def distribution():
    legal = np.random.rand(*DIM_MOVE)
    legal[legal > 0.8] = 1
    legal[legal < 0.9] = 0

    distr = np.random.rand(*DIM_MOVE)
    distr *= legal
    distr = distr.astype(np.float16)
    return distr


def outcome():
    x = np.random.rand(1)
    if x[0] > 0.6:
        return 1
    elif x[0] < 0.4:
        return -1
    else:
        return 0

def dataset(size = 100):
    data = []
    for i in range(size):
        data.append((position(), distribution(), outcome()))
    return data


if __name__ == "__main__":
    data = dataset()
    pickle.dump(data, open("dummy_data.pkl", "wb"))



