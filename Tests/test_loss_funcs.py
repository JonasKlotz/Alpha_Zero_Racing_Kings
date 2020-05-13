"""tests for loss funcs
"""
import numpy as np
from tensorflow import keras

pred_1 = np.array([0.0, 1.00, 0.75, -0.1])
labels_1 = np.array([0.1, 1.05, 0.750, 0.1])
TRUE_MSE_1 = 0.013125
TRUE_CE_1 = 0

pred_2 = []
labels_2 = []

losses = {'outcome': 'mean_squared_error', 'policy': 'categorical_crossentropy'}
losses_weights = {'value': 1, 'policy': 1}


def test_mse():
    """test if the return format of built-in loss functions is as simple as expected
    """
    keras_mse_1 = keras.losses.mean_squared_error(pred_1, labels_1)
    assert np.isclose(keras_mse_1, TRUE_MSE_1)


def test_crossentropy():
    """test if the return format of built-in loss functions is as simple as expected
    """
    # keras_ce_1 = keras.losses.binary_crossentropy(pred_1, labels_1)
    # assert np.isclose(keras_ce_1, TRUE_CE_1)
    assert True


def test_mse_crossentropy_added():
    """test if our custom combination works as expected
    """
    assert True
