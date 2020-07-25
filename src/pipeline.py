import json
import datetime
import requests
import numpy as np
import pandas as pd
import random

from src.save_parameters import save_to_json
from src.seirs_model import seirs_prediction
from src.loss_evaluation import *
from src.GeneticOptimizer import GeneticOptimizer
from seirsplus.models import SEIRSModel

random.seed(1138)
np.random.seed(1138)

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

param_ranges = {
    'beta': (0.0001, 2),  # Rate of transmission
    'sigma': (1 / 14, 1),  # Rate of progression
    'gamma': (1 / 10, 1),  # Rate of recovery
    'xi': (0.001, 0.001)  # Rate of re-susceptibility
}

# param_ranges_sir = {
#     'beta': (0.0001, 0.5),
#     'gamma': (0.0001, 0.5)
# }


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


# sir = SIR(USA_population, US_daily[day]['positive'].values[0], 0.1, 0.02)
# sir.run(T=15)
# real_positives = sir.total_num_infections()[10::10]

# optimizer = GeneticOptimizer(SIR,
#                              initI=US_daily[day]['positive'].values[0],
#                              initN=USA_population,
#                              param_ranges=param_ranges_sir,
#                              error_func=loss_function,
#                              real_values=real_positives,
#                              period=15,
#                              max_gen=1000,
#                              stop_cond=10000,
#                              # tournament_size=7,
#                              mut_range=0.02,
#                              p_regen=0.2,
#                              p_mut=0.4)

optimizer = GeneticOptimizer(SEIRSModel,
                             initI=US_daily[from_this_day_to_predict]['positive'].values[0],
                             initN=USA_population,
                             param_ranges=param_ranges,
                             error_func=mse_loss,
                             real_values=real_positives,
                             period=15,
                             max_gen=3000,
                             stop_cond=10000,
                             # tournament_size=7,
                             mut_range=0.1,
                             p_regen=0.2,
                             p_mut=0.4)

optimizer.initialize(population=100)
finished = False

best_counter = 0
current_best = None
while not finished and optimizer.g < optimizer.max_gen:
    finished, best = optimizer.iteration()
    if best != current_best:
        current_best = best
        best_counter = 0
    else:
        best_counter += 1
    if best_counter == 100:  # it can go on quite some time without changing the best fitness, depending on optimizer params
        finished = True

print(real_positives)

# infected_next_15_days = np.array(best)
# sir_pred = SIR(USA_population, US_daily[day]['positive'].values[0], **best)
# sir_pred.run(T=15)
# infected_next_15_days = np.array(sir_pred.total_num_infections()[10::10])

# Apply the model and optain a prediction for the next 15 days
infected_next_15_days = seirs_prediction(initI=US_daily[from_this_day_to_predict]['positive'].values[0],
                                         initN=USA_population,
                                         **best)

# # Apply the model and optain a prediction for the next 15 days
# infected_next_15_days = seirs_prediction(initI=US_daily[from_this_day_to_predict]['positive'],
#                                          initN=USA_population, **params)

print('Predictions from the next 15 days: ', [np.floor(x) for x in infected_next_15_days])

print('Real cases: ', real_positives)

mse_error = mse_loss(predicted_values=infected_next_15_days.reshape(-1, 1), real_values=real_positives)
mae_error = mae_loss(predicted_values=infected_next_15_days.reshape(-1, 1), real_values=real_positives)
# error_absolute_weights = custom_loss(predicted_values=infected_next_15_days.reshape(-1, 1),
#                                      real_values=real_positives,
#                                      sample_weight=[])

print('MSE: ', mse_error)
print('MAE: ', mae_error)
infected_next_15_days =map(int, infected_next_15_days)
real_positives = map(int, real_positives)
results = {'MSE': mse_error, 'MAE': mae_error, 'predictions_next_15_days': list(infected_next_15_days),
           'real_cases_15_days': list(real_positives)}

save_to_json(best, results)
# print('MAE weights: ', error_absolute_weights)

# print('2020-07-01 : ', US_daily[from_this_day_to_predict]['positive'])
