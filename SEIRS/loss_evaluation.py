from sklearn.metrics import mean_squared_error, mean_absolute_error


def loss_function(predicted_values, real_values):
    return mean_squared_error(predicted_values, real_values)

def custom_loss(predicted_values, real_values, sample_weight=None):
    return mean_absolute_error(predicted_values, real_values, sample_weight)


