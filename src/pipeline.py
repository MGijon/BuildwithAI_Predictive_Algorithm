import datetime
import numpy as np
import random

from src.api import get_us_daily, get_states_daily
from src.save_parameters import save_to_json
from src.seirs_model import seirs_prediction, seirs_prediction_with_a_lot_of_stuff
from src.loss_evaluation import mse_loss, mae_loss, weighted_mae_loss, weighted_mse_loss
from src.GeneticOptimizer import GeneticOptimizer
from seirsplus.models import SEIRSModel

random.seed(1138)
np.random.seed(1138)


class Predictor:
    # USA population according to a random internet source
    USA_population = 334_737_043

    def __init__(self, loss_days, init_date, state=None, param_ranges=None, genetic_params=None, states_data=None,
                 training=True, state_population=None):
        # Prefixed values
        self.loss_days = loss_days
        self.from_this_day_to_predict = init_date

        # API Call and data preparation
        if state is None:
            self.US_daily = get_us_daily()
            self.state = ""
        else:
            self.state_population = state_population
            if states_data is None:
                self.US_daily = get_states_daily()
            else:
                self.US_daily = states_data
            self.US_daily = self.US_daily[self.US_daily['state'] == state]
            self.US_daily.set_index('date', inplace=True)
            self.state = state

        # Initialization
        self.pred_positives = None
        if training:
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
                'xi': (0.0001, 0.0001)  # Rate of re-susceptibility
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
                                          initN=self.state_population if self.state else self.USA_population,
                                          param_ranges=param_ranges,
                                          error_func=weighted_mse_loss,
                                          real_values=self.real_positives,
                                          period=self.loss_days,
                                          **genetic_params)

        self.optimizer.initialize(population=100)
        self.finished = False

    def run(self, verbose=0):
        best_counter = 0
        current_best = None
        iterations = []
        while not self.finished and self.optimizer.g < self.optimizer.max_gen:
            self.finished, self.best = self.optimizer.iteration(verbose=verbose)
            iterations.append(self.best)
            if self.best != current_best:
                current_best = self.best
                best_counter = 0
            else:
                best_counter += 1
            # it can go on quite some time without changing the best fitness, depending on optimizer params
            if best_counter == 100:
                self.finished = True
        return iterations

    def get_pred_positives(self, date_to_start_predictions, number_of_days):
        assert self.best is not None
        # Apply the model and optain a prediction for the next 15 days
        infected_next_15_days = seirs_prediction(
            initI=self.US_daily[date_to_start_predictions]['positive'].values[0],
            initN=self.state_population if self.state else self.USA_population,
            initR=self.US_daily[date_to_start_predictions]['recovered'].values[0],
            predict_num_days=number_of_days,
            **self.best)

        pred_positives = list(infected_next_15_days.reshape(-1, ))

        return pred_positives

    def get_errors(self):
        predicted_in_validation_period = self.get_pred_positives(self.from_this_day_to_predict, self.loss_days)

        mse_error = mse_loss(predicted_values=predicted_in_validation_period, real_values=self.real_positives)
        mae_error = mae_loss(predicted_values=predicted_in_validation_period, real_values=self.real_positives)
        wmae_error = weighted_mae_loss(predicted_values=predicted_in_validation_period, real_values=self.real_positives)

        return mse_error, mae_error, wmae_error

    def report(self, date_to_start_predictions, number_of_days):
        predicted_in_validation_period = self.get_pred_positives(self.from_this_day_to_predict, self.loss_days)

        print("Real cases ({}): {}".format(self.from_this_day_to_predict, self.real_positives))
        print("Pred cases ({}): {}".format(self.from_this_day_to_predict, predicted_in_validation_period))

        prediced_in_future_period = self.get_pred_positives(date_to_start_predictions, number_of_days)

        print('Predictions from ({}) the next 15 days: '.format(date_to_start_predictions),
              [int(np.floor(x)) for x in prediced_in_future_period])

        mse_error, mae_error, wmae_error = self.get_errors()

        print('MSE: ', mse_error)
        print('MAE: ', mae_error)
        print('Weighted MAE: ', wmae_error)
        int_predicted_in_validation_period = map(int, predicted_in_validation_period)
        int_real_positives = map(int, self.real_positives)
        int_predicted_in_future_period = map(int, prediced_in_future_period)
        results = {'MSE': mse_error, 'MAE': mae_error, 'Weighted MAE': wmae_error,
                   'validation_predictions': list(
                       int_predicted_in_validation_period),
                   'real_cases': list(int_real_positives),
                   'final_prediction': list(int_predicted_in_future_period),
                   'validation_begin_date': self.from_this_day_to_predict,
                   'prediction_begin_date': date_to_start_predictions}

        to_save = {**self.best, **results}

        save_to_json(to_save, state=self.state)
        return to_save

    def generate_data_for_plots(self, date_to_start_predictions, number_of_days):
        prediced_s, prediced_e, prediced_i, prediced_r, prediced_f = seirs_prediction_with_a_lot_of_stuff(
            initI=self.US_daily[date_to_start_predictions]['positive'].values[0],
            initN=self.state_population if self.state else self.USA_population,
            initR=self.US_daily[date_to_start_predictions]['recovered'].values[0],
            predict_num_days=number_of_days,
            **self.best)
        prediced_s = list(map(int, prediced_s))
        prediced_e = list(map(int, prediced_e))
        prediced_i = list(map(int, prediced_i))
        prediced_r = list(map(int, prediced_r))
        prediced_f = list(map(int, prediced_f))

        for_saving = {'S': prediced_s, 'E': prediced_e, 'I': prediced_i, 'R': prediced_r, 'F': prediced_f}
        return for_saving
