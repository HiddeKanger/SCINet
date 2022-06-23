import numpy as np
import matplotlib.pyplot as plt

def orig_values(orig_maes,datasets):
    values = []
    for dataset in datasets:
        values.append(orig_maes[dataset])
    return values

def plot_barplot_two(datasets, orig_maes, maes):

    add_space = np.arange(len(datasets))*0.4

    plt.bar(np.arange(0,len(datasets))+add_space,orig_values(orig_maes,datasets),width = 0.5, align='edge', color = 'blue', edgecolor = 'black', label = 'Original')
    plt.bar(np.arange(0,len(datasets))+0.5+add_space,maes,align='edge',width = 0.5, color = 'lightblue', edgecolor = 'black', label = 'Tensorflow implementation')
    plt.xticks(np.arange(0,len(datasets))+0.5+add_space, datasets, rotation = 30, fontsize = 14)
    plt.ylabel('Mean absolute error')
    plt.legend()
    plt.title('Comparison original and Tensorflow implementation')

    plt.savefig('exp/reprod/results/comparison.pdf')


def plot_barplot_three(datasets, orig_maes, maes, maes_leaky):

    add_space = np.arange(len(datasets))*0.4

    plt.bar(np.arange(0,len(datasets))+add_space,orig_values(orig_maes,datasets),width = 1/3, align='edge', color = 'orange', edgecolor = 'black', label = 'Original')
    plt.bar(np.arange(0,len(datasets))+1/3+add_space,maes,align='edge',width = 1/3, color = 'blue', edgecolor = 'black', label = 'Tensorflow implementation')
    plt.bar(np.arange(0,len(datasets))+2/3+add_space,maes_leaky,align='edge',width = 1/3, color = 'lightblue', edgecolor = 'black', label = 'Tensorflow leaky implementation')
    plt.xticks(np.arange(0,len(datasets))+0.5+add_space, datasets, rotation = 30, fontsize = 14)
    plt.ylabel('Mean absolute error')
    plt.legend()
    plt.title('Comparison original and Tensorflow implementation')

    plt.savefig('exp/reprod/results/comparison_leaky.pdf')