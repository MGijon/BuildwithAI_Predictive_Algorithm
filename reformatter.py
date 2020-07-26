from os import listdir
from os.path import isfile, join
import json

path = "./results/"
output_file = "states_summary.json"
files = [f for f in listdir(path) if isfile(join(path, f))]

report_flag = True
us_predictions = list()
states = dict()
for f in files:
    if report_flag and "report_beta" in f:
        report_flag = False

        with open("{}{}".format(path, f), "r") as json_file:
            content = json.loads(json_file.read())
            us_predictions = content['final_prediction']
    elif "report_" in f:
        tmp = f.replace("report_", "")
        state = tmp[:tmp.find("_")]
        with open("{}{}".format(path, f), "r") as json_file:
            content = json.loads(json_file.read())
            states[state] = content['final_prediction']

output = {"us": us_predictions}
for state in states:
    output[state] = states[state]

with open("{}{}".format(path, output_file), "w+") as json_file:
    json.dump(output, json_file, sort_keys=False, indent=4)

