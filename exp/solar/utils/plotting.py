import numpy as np

import matplotlib.pyplot as plt


def plot_histograms(n_columns, data):

    fig, axes = plt.subplots(data.shape[2]//n_columns,n_columns, figsize = (20,4))

    for i in range(data.shape[2]):

        axes[i//8,i%8].hist((data[:,:,i].flatten()), bins = 40)
        axes[i//8,i%8].ticklabel_format(axis = 'y', scilimits = (-2,2))
        axes[i//8,i%8].set_title('Feature {}'.format(i+1))
        axes[i//8,i%8].grid()
        axes[1, i%8].set_xlabel('value in (a.u.)')
        axes[i//8, 0].set_ylabel('Frequency')

        axes[0,0].set_title('Flux')

    plt.tight_layout()

def barplot_correlations(n_columns, mean_correlations):

    fig, ax = plt.subplots(1,1, figsize = (6,4))

    ax.bar(np.arange(1,n_columns+1), mean_correlations, align = 'center', width = 1, color = 'blue', edgecolor = 'black')
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(1,n_columns+1))
    ax.set_ylabel('Mean Correlation coefficient')
    ax.set_xlabel('Feature number')

    fig.suptitle('Correlation between first column and consequent columns')