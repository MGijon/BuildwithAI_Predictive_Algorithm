import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mse_loss(predicted_values, real_values):
    return mean_squared_error(predicted_values, real_values)


def mae_loss(predicted_values, real_values):
    return mean_absolute_error(predicted_values, real_values)


def weighted_mae_loss(predicted_values, real_values):
    return mean_absolute_error(predicted_values, real_values,
                               sample_weight=np.power(np.linspace(0.0, 1.0, num=36), 2)[1:])


def weighted_mse_loss(predicted_values, real_values):
    return mean_squared_error(predicted_values, real_values,
                              sample_weight=np.power(np.linspace(0.0, 1.0, num=36), 2)[1:])
