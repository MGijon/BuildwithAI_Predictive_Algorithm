import json
import datetime as dt
import requests
import numpy as np
import pandas as pd
import bayes_opt as bo

from sklearn.metrics import mean_squared_error, mean_absolute_error
from seirsplus.models import SEIRSModel


def seirs_prediction(initI, initN, T, **params):
    model = SEIRSModel(initN=initN, initI=initI, **params)
    model.run(T=T, verbose=False)
    return model.total_num_infections()[10::10]

def loss_function(predicted_values, real_values):
    return mean_squared_error(predicted_values, real_values)

def custom_loss(predicted_values, real_values, sample_weight=None):
    return mean_absolute_error(predicted_values, real_values, sample_weight=sample_weight)