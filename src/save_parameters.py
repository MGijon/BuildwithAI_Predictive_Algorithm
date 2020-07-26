import json
import os


def save_to_json(values, path=None, file_name=None, state=""):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), os.pardir, 'results')
    os.makedirs(path, exist_ok=True)
    num_of_experiments = len(os.listdir(path))
    if file_name is None:
        file_name = "experiment_{}_{}".format(state, num_of_experiments)
    else:
        file_name = file_name

    with open(os.path.join(path, file_name), 'w+') as json_file:
        json.dump(values, json_file, sort_keys=False, indent=4)
