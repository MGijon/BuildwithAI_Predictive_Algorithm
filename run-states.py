import json

from src.pipeline import Predictor

# STATES = ['NY', 'LA']
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


for state in STATES:
    predictor = Predictor(loss_days=15, init_date='2020-07-01', state=state, param_ranges=param_ranges,
                          genetic_params=genetic_params)
    iterations = predictor.run()

    with open('results/iterations_{}.json'.format(state), 'w+') as json_file:
        json.dump(iterations, json_file, sort_keys=True, indent=4)

    predictor.report()
