import json
import os

from src.pipeline import Predictor
from src.plot_to_gif import generate_gif_from_iterations_for_the_seir_parameters, generate_seir_gif
from src.save_parameters import save_to_json

PATH_OF_FILES_TO_PLOT = 'results/to_plot'
OUTPUT_PATH = 'results/plots'


def load_from_json(path):
    with open(path, 'r') as file:
        values = json.load(file)
    return values


for file_name in os.listdir(PATH_OF_FILES_TO_PLOT):
    file_path = os.path.join(PATH_OF_FILES_TO_PLOT, file_name)
    data = load_from_json(file_path)
    if 'seir' in file_name:
        print('ploting seir to: {}'.format(OUTPUT_PATH + file_name.replace('json', 'gif')))
        generate_seir_gif(data, path=OUTPUT_PATH, name=file_name.replace('json', 'gif'))
    elif 'iterations' in file_name:
        print('ploting iterations to: {}'.format(OUTPUT_PATH + file_name.replace('json', 'gif')))
        generate_gif_from_iterations_for_the_seir_parameters(data, path=OUTPUT_PATH,
                                                             name=file_name.replace('json', 'gif'))
