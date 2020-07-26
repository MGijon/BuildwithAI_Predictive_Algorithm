import datetime
import json
import numpy as np

from src.api import get_states_daily, fill_data
from src.loss_evaluation import mean_absolute_error, weighted_mae_loss
from src.pipeline import Predictor

# STATES = ['AK']
NUMBER_OF_DAYS_TRAINING = 45
NUMBER_OF_DAYS_PREDICTING = 25

BEGIN_DATE_TRAINING = '2020-06-10'
BEGIN_DATE_PREDICTING = '2020-07-25'


# STATES = ['AS']
STATES = ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN',
          'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
          'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI', 'VT', 'WA', 'WI',
          'WV', 'WY']

STATE_POPULATIONS = {
    'AK': 734002,
    'AL': 4908620,
    'AR': 3039000,
    'AS': 55212,
    'AZ': 7378490,
    'CA': 39937500,
    'CO': 5845530,
    'CT': 3563080,
    'DC': 720687,
    'DE': 982895,
    'FL': 21993000,
    'GA': 10736100,
    'GU': 168485,
    'HI': 1412690,
    'IA': 3179850,
    'ID': 1826160,
    'IL': 12659700,
    'IN': 6745350,
    'KS': 2910360,
    'KY': 4499690,
    'LA': 4645180,
    'MA': 6976600,
    'MD': 6083120,
    'ME': 1345790,
    'MI': 10045000,
    'MN': 5700670,
    'MO': 6169270,
    'MP': 57581,
    'MS': 2989260,
    'MT': 1086760,
    'NC': 10611900,
    'ND': 761723,
    'NE': 1952570,
    'NH': 1371250,
    'NJ': 8936570,
    'NM': 2096640,
    'NV': 3139660,
    'NY': 19440500,
    'OH': 11747700,
    'OK': 3954820,
    'OR': 4301090,
    'PA': 12820900,
    'PR': 3032160,
    'RI': 1056160,
    'SC': 5210100,
    'SD': 903027,
    'TN': 6897580,
    'TX': 29472300,
    'UT': 3282120,
    'VA': 8626210,
    'VI': 104425,
    'VT': 628061,
    'WA': 7797100,
    'WI': 5851750,
    'WV': 1778070,
    'WY': 567025
}

param_ranges = {
                'beta': (0.0001, 2),  # Rate of transmission
                'sigma': (1 / 14, 1),  # Rate of progression
                'gamma': (1 / 10, 1),  # Rate of recoveryrecovery
                'mu_I': (0.0001, 1 / 10),  # Rate of DEATH
                'xi': (0.0001, 0.0001)  # Rate of re-susceptibility
            }

genetic_params = {
                  'max_gen': 30,
                  'stop_cond': 10000,
                  'mut_range': 0.1,
                  'p_regen': 0.2,
                  'p_mut': 0.4
            }

us_results_training = {
    'S': np.zeros(45, dtype=np.float),
    'E': np.zeros(45, dtype=np.float),
    'I': np.zeros(45, dtype=np.float),
    'R': np.zeros(45, dtype=np.float),
    'F': np.zeros(45, dtype=np.float)
}

us_results_predicting = {
    'S': np.zeros(45, dtype=np.float),
    'E': np.zeros(45, dtype=np.float),
    'I': np.zeros(45, dtype=np.float),
    'R': np.zeros(45, dtype=np.float),
    'F': np.zeros(45, dtype=np.float)
}

data = get_states_daily()
data = fill_data(data, '2020-07-25')
for state in STATES:
    print("Predicting for {}...".format(state))
    predictor = Predictor(loss_days=NUMBER_OF_DAYS_TRAINING, init_date=BEGIN_DATE_TRAINING, state=state, param_ranges=param_ranges,
                          genetic_params=genetic_params, states_data=data, state_population=STATE_POPULATIONS[state])
    iterations = predictor.run()

    state_results = predictor.generate_data_for_plots(BEGIN_DATE_TRAINING, NUMBER_OF_DAYS_TRAINING)
    for n in state_results:
        us_results_training[n] += np.array(state_results[n])

    with open('results/iterations_{}.json'.format(state), 'w+') as json_file:
        json.dump(iterations, json_file, sort_keys=True, indent=4)

    seir_data = predictor.generate_data_for_plots(BEGIN_DATE_PREDICTING, NUMBER_OF_DAYS_PREDICTING)

    with open('results/seir_{}.json'.format(state), 'w+') as json_file:
        json.dump(seir_data, json_file, sort_keys=True, indent=4)

    # predictor.report()
    print("Done!..")

us_predictor = Predictor(loss_days=NUMBER_OF_DAYS_TRAINING, init_date=BEGIN_DATE_TRAINING, param_ranges=param_ranges, genetic_params=genetic_params)
real_data = us_predictor.US_daily

start = datetime.datetime.strptime(us_predictor.from_this_day_to_predict, '%Y-%m-%d')
start = start + datetime.timedelta(days=1)

time_delta = datetime.timedelta(days=us_predictor.loss_days - 1)
end = start + time_delta

real_positives = []
real_recovered = []

step = datetime.timedelta(days=1)

while start <= end:
    day = start.strftime('%Y-%m-%d')
    real_positives.append(int(us_predictor.US_daily[day]['positive'].values[0]))  # date()
    real_recovered.append(int(us_predictor.US_daily[day]['recovered'].values[0]))
    start += step

print("Predicted infected: {}\nReal infected: {}\n\nPredicted recovered: {}\nTrue recovered: {}".format(us_results_training['I'],
                                                                                                        real_positives,
                                                                                                        us_results_training['R'],
                                                                                                        real_recovered))
print("MAE for merged data: {}\nWeighted MAE for merged data: {}".format(mean_absolute_error(real_positives, us_results_training['I']),
                                                                         weighted_mae_loss(us_results_training['I'], real_positives)))

with open("results/prediction_states.json", "w+") as json_file:
    json.dump({"S": list(us_results_training["S"]),
               "E": list(us_results_training["E"]),
               "I": list(us_results_training["I"]),
               "R": list(us_results_training["R"]),
               "F": list(us_results_training["F"]),
               "real_I": list(real_positives),
               "real_R": list(real_recovered)},
              json_file)
