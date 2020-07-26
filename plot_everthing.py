import json
import os

from src.plot_to_gif import generate_gif_from_iterations_for_the_seir_parameters, generate_seir_gif, plot_prediction, \
    plot_prediction_state

PATH_OF_FILES_TO_PLOT = 'results/to_plot'
OUTPUT_PATH = 'results/plots'


def load_from_json(path):
    with open(path, 'r') as file:
        values = json.load(file)
    return values


states = ['AK', 'AL', 'AR', 'AS', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'GU', 'HI', 'IA', 'ID', 'IL',
          'IN',
          'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MP', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ',
          'NM',
          'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI', 'VT', 'WA',
          'WI',
          'WV', 'WY']

for file_name in os.listdir(PATH_OF_FILES_TO_PLOT):
    file_path = os.path.join(PATH_OF_FILES_TO_PLOT, file_name)
    data = load_from_json(file_path)
    if 'seir' in file_name:

        print('ploting seir to: {}'.format(OUTPUT_PATH + '/' + file_name.replace('json', 'gif')))
        generate_seir_gif(data, path=OUTPUT_PATH, name=file_name.replace('json', 'gif'))
    elif 'iterations' in file_name:

        print('ploting iterations to: {}'.format(OUTPUT_PATH + '/' + file_name.replace('json', 'gif')))
        generate_gif_from_iterations_for_the_seir_parameters(data, path=OUTPUT_PATH,
                                                             name=file_name.replace('json', 'gif'))
    elif 'report' in file_name:
        print('ploting report to: {}'.format(OUTPUT_PATH + '/' + file_name.replace('json', 'gif')))
        if len(file_name.split('_')) > 2:
            plot_prediction(data, path=OUTPUT_PATH,
                            name=file_name.replace('json', 'gif'))
        else:
            state = file_name.split('.')[0].split('_')[1]
            plot_prediction_state(data, state, path=OUTPUT_PATH,
                                  name=file_name.replace('json', 'gif'))
