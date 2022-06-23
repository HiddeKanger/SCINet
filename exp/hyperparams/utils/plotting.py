import numpy as np
import matplotlib.pyplot as plt

def orig_values(maes,hyperparam_settings):
    values = []
    for setting in range(0,len(hyperparam_settings)):
        values.append(maes[setting][0])
    print(values)
    return values

def plot_barplot(hyperparam_settings, maes, hyperparameter_type):

    add_space = np.arange(len(hyperparam_settings))*0.4

    plt.bar(np.arange(0,len(hyperparam_settings))+add_space,orig_values(maes,hyperparam_settings),width = 0.5, align='edge', color = 'blue', edgecolor = 'black')
    #plt.bar(np.arange(0,len(datasets))+0.5+add_space,maes,align='edge',width = 0.5, color = 'lightblue', edgecolor = 'black', label = 'Tensorflow implementation')
    plt.xticks(np.arange(0,len(hyperparam_settings))+0.5+add_space, hyperparam_settings, rotation = 30, fontsize = 14)
    plt.ylabel('Validation Mean absolute error')
    plt.legend()
    plt.title(f'Comparison of {hyperparameter_type}')
    plt.savefig(f"results/Optimization_{hyperparameter_type}.pdf")
    plt.show()