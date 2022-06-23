import numpy as np
import matplotlib.pyplot as plt


def plot_barplot(type, values, maes, ):

    plt.bar(np.arange(0,len(values)),maes,width = 0.5, align='center', color = 'blue', edgecolor = 'black')
    plt.xticks(np.arange(0,len(values)), values, rotation = 30, fontsize = 14)
    plt.ylabel('Validation Mean absolute error')
    plt.legend()
    plt.title(f'Comparison of {type}')
    plt.savefig(f"results/Optimization_{type}.pdf")
    plt.show()
