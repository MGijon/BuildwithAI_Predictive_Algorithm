import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.animation import FuncAnimation
import seaborn as sns

sns.set()


def plot_prediction():
    pass


def generate_gif_from_iterations_for_the_seir_parameters(params, path='results/plots', name='beta_gamma_sigma',
                                                         frames_to_save=None):
    params = pd.DataFrame(params)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.set_tight_layout(True)
    plt.xlim(0, params.shape[0])
    plt.ylim(0, 1)

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

    anim = FuncAnimation(fig, update, frames=np.arange(0, frames_to_save, 10), interval=200)
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
