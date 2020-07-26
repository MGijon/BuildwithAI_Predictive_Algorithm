import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.animation import FuncAnimation
import seaborn as sns

sns.set()


def generate_gif_from_iterations_for_the_seir_parameters(params, path='results', name='beta_gamma_sigma', frames_to_save=None):
    params = pd.DataFrame(params)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.set_tight_layout(True)
    plt.xlim(0, params.shape[0])
    plt.ylim(0, 1)

    palette = sns.color_palette()

    if frames_to_save is None:
        frames_to_save = params.shape[0]

    def update(i):
        label = 'Generation {0}'.format(i)
        data = params.iloc[:int(i + 1)]  # select data range
        sns.lineplot(x=data.index, y=data['beta'], data=data, label='beta', color=palette[0], ax=ax, legend=False)
        sns.lineplot(x=data.index, y=data['gamma'], data=data, label='gamma', color=palette[1], ax=ax, legend=False)
        sns.lineplot(x=data.index, y=data['sigma'], data=data, label='sigma', color=palette[2], ax=ax, legend=False)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[:3], labels[:3])

        ax.set_ylabel('SEIR parameters')
        ax.set_xlabel(label)
        return ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, frames_to_save), interval=200)
    anim.save(os.path.join(path, '{}.gif'.format(name)), dpi=80, writer='imagemagick')
