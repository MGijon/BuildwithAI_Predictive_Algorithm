import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.animation import FuncAnimation
import seaborn as sns

from src.api import get_us_daily, get_states_daily

sns.set()


def plot_prediction(results, begin_plot=True, path='results/plots', name='prediction_plot'):
    us_data = get_us_daily()
    us_data = us_data[['positive']]
    us_data['origin'] = 'Real Data'

    prediction = results['final_prediction']

    start = datetime.datetime.strptime(results['prediction_begin_date'], '%Y-%m-%d')
    start = start + datetime.timedelta(days=1)

    new_dates = pd.date_range(start=start, periods=len(prediction))

    predicted_data = pd.DataFrame(prediction,
                                  index=new_dates,
                                  columns=['positive'])
    predicted_data['origin'] = 'Predicted Results'

    us_data_with_prediction = us_data.append(predicted_data).sort_index()
    if begin_plot:
        begin_plpt_date = results['validation_begin_date']
        us_data_with_prediction = us_data_with_prediction[begin_plpt_date:]
    plt.figure(figsize=(12, 4))
    original_values = us_data_with_prediction['origin'] == 'Real Data'
    predicted_values = us_data_with_prediction['origin'] == 'Predicted Results'
    sns.lineplot(data=us_data_with_prediction.loc[original_values, 'positive'], label='Real data')
    sns.lineplot(data=us_data_with_prediction.loc[predicted_values, 'positive'], label='Predicted data')
    plt.xticks(rotation=90)
    plt.title('Positive cases in US')
    plt.savefig(os.path.join(path, '{}.png'.format(name)), dpi=200, bbox_inches='tight')


def plot_prediction_state(results, state, begin_plot=True, path='results/plots', name='prediction_plot'):
    states = get_states_daily()

    target_state = states[states['state'] == state].drop(columns=['state']).set_index('date')
    us_data = target_state[['positive']]
    us_data['origin'] = 'Real Data'

    prediction = results['final_prediction']

    start = datetime.datetime.strptime(results['prediction_begin_date'], '%Y-%m-%d')
    start = start + datetime.timedelta(days=1)

    new_dates = pd.date_range(start=start, periods=len(prediction))

    predicted_data = pd.DataFrame(prediction,
                                  index=new_dates,
                                  columns=['positive'])
    predicted_data['origin'] = 'Predicted Results'

    us_data_with_prediction = us_data.append(predicted_data).sort_index()
    if begin_plot:
        begin_plpt_date = results['validation_begin_date']
        us_data_with_prediction = us_data_with_prediction[begin_plpt_date:]
    plt.figure(figsize=(12, 4))
    original_values = us_data_with_prediction['origin'] == 'Real Data'
    predicted_values = us_data_with_prediction['origin'] == 'Predicted Results'
    sns.lineplot(data=us_data_with_prediction.loc[original_values, 'positive'], label='Real data')
    sns.lineplot(data=us_data_with_prediction.loc[predicted_values, 'positive'], label='Predicted data')
    plt.xticks(rotation=90)
    plt.title('Positive cases in {}'.format(state))
    plt.savefig(os.path.join(path, '{}.png'.format(name)), dpi=200, bbox_inches='tight')


def generate_gif_from_iterations_for_the_seir_parameters(params, path='results/plots', name='beta_gamma_sigma',
                                                         frames_to_save=None):
    params = pd.DataFrame(params)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.set_tight_layout(True)
    plt.xlim(0, params.shape[0])
    plt.ylim(0, 2)

    palette = sns.color_palette()

    if frames_to_save is None:
        frames_to_save = params.shape[0]

    def update(i):
        label = 'Generation {}'.format(i)
        data = params.iloc[:int(i + 1)]  # select data range
        for i, column in enumerate(data.columns):
            sns.lineplot(x=data.index, y=data[column], data=data, label=column, color=palette[i], ax=ax, legend=False)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[:i], labels[:i])

        ax.set_ylabel('SEIR parameters')
        ax.set_xlabel(label)
        return ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, frames_to_save, 30), interval=200)
    anim.save(os.path.join(path, '{}.gif'.format(name)), dpi=200, writer='imagemagick')


def generate_seir_gif(seir, path='results/plots', name='seir', ):
    seir = pd.DataFrame(seir)

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.set_tight_layout(True)

    palette = sns.color_palette()

    def update(i):
        labels = seir.columns
        sizes = seir.iloc[i].values
        g = sns.barplot(x=labels, y=sizes, order=list('SEIRF'))
        g.set_yscale("log")
        plt.title('SEIR values at day {}'.format(i))
        ax.set_ylabel('log scale')

        return ax

    plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=np.arange(0, seir.shape[0]), interval=200)
    anim.save(os.path.join(path, '{}.gif'.format(name)), dpi=200, writer='imagemagick')
