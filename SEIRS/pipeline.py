import json
import datetime
import requests
import numpy as np
import pandas as pd

from seirs_model import seirs_prediction
from loss_evaluation import *

#### PREFIXED VALUES ####
from_this_day_to_predict = '2020-07-01'  # later to change for the '2020-07-27'
USA_population = 328_200_000  # USA population according to a random internet source

# API Call and data preparation
payload = {}
headers = {}

url = "https://covidtracking.com/api/us/daily"
response_US_Daily = requests.request("GET", url, headers=headers, data=payload)

US_daily = pd.DataFrame(json.loads(response_US_Daily.text))

US_daily['date'] = pd.to_datetime(US_daily['date'], format='%Y%m%d')
US_daily.set_index('date', inplace=True)

#### PREDICTION ####
####################

# TO OPTIMIZE WITH MAGIC, SORCERY AND ENCHANTMENTS
params = {
    'beta': 0.155,  # Rate of transmission
    'sigma': 1 / 5.2,  # Rate of progression
    'gamma': 1 / 12.39,  # Rate of recovery
    'xi': 0.001  # Rate of re-susceptibility
}

# Apply the model and optain a prediction for the next 15 days
infected_next_15_days = seirs_prediction(initI=US_daily[from_this_day_to_predict]['positive'],
                                         initN=USA_population, **params)

print('Predictions from the next 15 days: ', [np.floor(x) for x in infected_next_15_days])

#### ERRORS ####
################
start = datetime.datetime.strptime(from_this_day_to_predict, '%Y-%m-%d')
start = start + datetime.timedelta(days=1)

time_delta = datetime.timedelta(days=14)
end = start + time_delta

real_positives = []

step = datetime.timedelta(days=1)

while start <= end:
    day = start.strftime('%Y-%m-%d')
    real_positives.append(US_daily[day]['positive'].values[0])  # date()
    start += step

print('Real cases: ', real_positives)

errors = loss_function(predicted_values=infected_next_15_days, real_values=real_positives)
error_absolute = custom_loss(predicted_values=infected_next_15_days, real_values=real_positives)
error_absolute_weights = custom_loss(predicted_values=infected_next_15_days,
                                     real_values=real_positives,
                                     sample_weight=[])

print('MSE: ', errors)
print('MAE: ', error_absolute)
print('MAE weights: ', error_absolute_weights)

#print('2020-07-01 : ', US_daily[from_this_day_to_predict]['positive'])