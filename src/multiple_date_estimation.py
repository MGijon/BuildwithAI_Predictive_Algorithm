from src.pipeline import Predictor


def params_for_date(date):
    print('Computing params for {}...'.format(date))
    predictor = Predictor(loss_days=15, init_date=date)
    predictor.run()

    params = predictor.best
    errors = predictor.get_errors()
    errors_dict = {name: error for name, error in zip(['MSE', 'MAE', 'Weighted MAE'], errors)}

    print("Best params: {}".format(params))
    print("Errors: {}".format(errors_dict))
    return params, errors


dates_to_eval = [
    '2020-07-10',
    '2020-07-03',
    '2020-06-26',
    '2020-06-19',
    '2020-06-12',
]
weekly_params = {date: {} for date in dates_to_eval}
for date in dates_to_eval:
    weekly_params[date] = params_for_date(date)

print(weekly_params)
