import datetime
import json
import numpy as np

from src.api import get_states_daily, fill_data
from src.loss_evaluation import mean_absolute_error, weighted_mae_loss
from src.pipeline import Predictor

# STATES = ['NY']
STATES = ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL', 'IN',
          'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
          'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI', 'VT', 'WA', 'WI',
          'WV', 'WY']

param_ranges = {
                'beta': (0.0001, 2),  # Rate of transmission
                'sigma': (1 / 14, 1),  # Rate of progression
                'gamma': (1 / 10, 1),  # Rate of recovery
                'xi': (0.0001, 0.0001)  # Rate of re-susceptibility
            }

genetic_params = {
                  'max_gen': 3000,
                  'stop_cond': 10000,
                  'mut_range': 0.1,
                  'p_regen': 0.2,
                  'p_mut': 0.4
            }

us_results = {
    'S': np.zeros(15, dtype=np.float),
    'E': np.zeros(15, dtype=np.float),
    'I': np.zeros(15, dtype=np.float),
    'R': np.zeros(15, dtype=np.float),
    'F': np.zeros(15, dtype=np.float)
}

data = get_states_daily()
data = fill_data(data, '2020-07-25')
for state in STATES:
    print("Predicting for {}...".format(state))
    predictor = Predictor(loss_days=15, init_date='2020-07-01', state=state, param_ranges=param_ranges,
                          genetic_params=genetic_params, states_data=data)
    iterations = predictor.run()

    state_results = predictor.generate_data_for_plots()
    for n in state_results:
        us_results[n] += np.array(state_results[n])

    with open('results/iterations_{}.json'.format(state), 'w+') as json_file:
        json.dump(iterations, json_file, sort_keys=True, indent=4)

    # predictor.report()
    print("Done!..")

us_predictor = Predictor(loss_days=15, init_date='2020-07-01', param_ranges=param_ranges, genetic_params=genetic_params)
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

print("Predicted infected: {}\nReal infected: {}\n\nPredicted recovered: {}\nTrue recovered: {}".format(us_results['I'],
                                                                                                        real_positives,
                                                                                                        us_results['R'],
                                                                                                        real_recovered))
print("MAE for merged data: {}\nWeighted MAE for merged data: {}".format(mean_absolute_error(real_positives, us_results['I']),
                                                                         weighted_mae_loss(us_results['I'], real_positives)))

with open("results/prediction_states", "w+") as json_file:
    json.dump({"pred_I": list(us_results["I"]), "real_I": list(real_positives), "pred_R": list(us_results["R"]),
               "real_R": list(real_recovered)}, json_file)
