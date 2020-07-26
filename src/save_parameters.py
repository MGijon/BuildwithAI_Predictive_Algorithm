import json
import os


def save_to_json(params, args, path=None, state=""):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), os.pardir, 'results')
    os.makedirs(path, exist_ok=True)
    params_with_args = {**params, **args}
    num_of_experiments = len(os.listdir(path))
    if 'MAE' in args.keys():
        file_name = "experiment_{}_{}_{}".format(state, num_of_experiments, args['MAE'])
    else:
        file_name = "experiment_{}_{}".format(state, num_of_experiments)
    with open(os.path.join(path, file_name), 'w+') as json_file:
        json.dump(params_with_args, json_file, sort_keys=True, indent=4)

