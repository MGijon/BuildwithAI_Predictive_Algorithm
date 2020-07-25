from sklearn.metrics import mean_squared_error


def loss_function(predicted_values, real_values):

    return mean_squared_error(predicted_values, real_values)