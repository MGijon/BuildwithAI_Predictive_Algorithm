import datetime
import numpy as np
import random

from src.api import get_us_daily
from src.save_parameters import save_to_json
from src.seirs_model import seirs_prediction
from src.loss_evaluation import mse_loss, mae_loss
from src.GeneticOptimizer import GeneticOptimizer
from seirsplus.models import SEIRSModel

random.seed(1138)
np.random.seed(1138)


class Predictor:
    # USA population according to a random internet source
    USA_population = 328_200_000

    def __init__(self, loss_days, init_date, param_ranges=None, genetic_params=None):
        # Prefixed values
        self.loss_days = loss_days
        self.from_this_day_to_predict = init_date

        # API Call and data preparation
        self.US_daily = get_us_daily()

        # Initialization
        self.real_positives = self.get_real_data()
        self.optimizer = None
        self.finished = None
        self.best = None
        self._init_optimizer(param_ranges, genetic_params)

    def get_real_data(self):
        start = datetime.datetime.strptime(self.from_this_day_to_predict, '%Y-%m-%d')
        start = start + datetime.timedelta(days=1)

        time_delta = datetime.timedelta(days=self.loss_days - 1)
        end = start + time_delta

        real_positives = []

        step = datetime.timedelta(days=1)

        while start <= end:
            day = start.strftime('%Y-%m-%d')
            real_positives.append(self.US_daily[day]['positive'].values[0])  # date()
            start += step

        return real_positives

    def _init_optimizer(self, param_ranges, genetic_params):
        if not param_ranges:
            param_ranges = {
                'beta': (0.0001, 2),  # Rate of transmission
                'sigma': (1 / 14, 1),  # Rate of progression
                'gamma': (1 / 10, 1),  # Rate of recovery
                'xi': (0.001, 0.001)  # Rate of re-susceptibility
            }
        if not genetic_params:
            genetic_params = {
                  'max_gen': 3000,
                  'stop_cond': 10000,
                  'mut_range': 0.1,
                  'p_regen': 0.2,
                  'p_mut': 0.4
            }
        self.optimizer = GeneticOptimizer(SEIRSModel,
                                          initI=self.US_daily[self.from_this_day_to_predict]['positive'].values[0],
                                          initR=self.US_daily[self.from_this_day_to_predict]['recovered'].values[0],
                                          initN=self.USA_population,
                                          param_ranges=param_ranges,
                                          error_func=mse_loss,
                                          real_values=self.real_positives,
                                          period=self.loss_days,
                                          **genetic_params)

        self.optimizer.initialize(population=100)
        self.finished = False

    def run(self):
        best_counter = 0
        current_best = None
        while not self.finished and self.optimizer.g < self.optimizer.max_gen:
            self.finished, self.best = self.optimizer.iteration()
            if self.best != current_best:
                current_best = self.best
                best_counter = 0
            else:
                best_counter += 1
            # it can go on quite some time without changing the best fitness, depending on optimizer params
            if best_counter == 100:
                self.finished = True

    def report(self):
        print(self.real_positives)

        # Apply the model and optain a prediction for the next 15 days
        infected_next_15_days = seirs_prediction(
            initI=self.US_daily[self.from_this_day_to_predict]['positive'].values[0],
            initN=self.USA_population,
            **self.best)

        print('Predictions from the next 15 days: ', [np.floor(x) for x in infected_next_15_days])

        print('Real cases: ', self.real_positives)

        mse_error = mse_loss(predicted_values=infected_next_15_days.reshape(-1, 1), real_values=self.real_positives)
        mae_error = mae_loss(predicted_values=infected_next_15_days.reshape(-1, 1), real_values=self.real_positives)
        # error_absolute_weights = custom_loss(predicted_values=infected_next_15_days.reshape(-1, 1),
        #                                      real_values=real_positives,
        #                                      sample_weight=[])

        print('MSE: ', mse_error)
        print('MAE: ', mae_error)
        infected_next_15_days = map(int, infected_next_15_days)
        real_positives = map(int, self.real_positives)
        results = {'MSE': mse_error, 'MAE': mae_error, 'predictions_next_15_days': list(infected_next_15_days),
                   'real_cases_15_days': list(real_positives)}

        save_to_json(self.best, results)
