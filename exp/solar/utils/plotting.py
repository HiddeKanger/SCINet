import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as mae


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

def plot_model_predictions(X_test_proc,y_test_proc,predictions,samples,\
    N_rows,N_columns,X_len,Y_len, columns_string):

    skip_x = 50
    time_x = np.arange(skip_x,X_len)
    time_y = np.arange(X_len,X_len+Y_len)

    colors = ['blue','red','green','maroon']

    fig, axes = plt.subplots(N_rows, N_columns, figsize = (20,8), sharex= True)

    for i, sample in enumerate(samples):

        row = i//N_columns
        col = i%N_columns

        axes[row, col].plot(time_x, X_test_proc[sample,skip_x:,0], color = 'black', label = 'SCINet input')
        axes[row, col].plot(time_y, y_test_proc[sample,:,0], color = 'black', ls = '--', label = 'Real outputs')

        for j, prediction in enumerate(predictions):

            axes[row, col].plot(time_y, prediction[sample,:], color = colors[j], ls = '--', \
                        label = 'used cols: [' + columns_string[j] + ']')

        axes[0,0].legend(ncol = 6, loc =0, bbox_to_anchor = (4.7,1.2,0,0), fontsize = 14)
        axes[row,col].grid()
        axes[N_rows-1,col].set_xlabel('time in (a.u.)')
        axes[row,0].set_ylabel('normalised flux in (a.u.)')

    fig.suptitle('Example predictions made by trained models', fontsize = 22)

def plot_per_timestep_mae(truths,predictions, constant_predictions, Y_len, labels):

    time_from_last_x = np.arange(1,Y_len+1)
    time_wise_mae_constant = []
    time_wise_mae = [ [] for _ in range(len(predictions)) ]

    colors = ['blue','red','green','maroon']

    for i in range(truths.shape[1]):

        time_wise_mae_constant.append(mae(truths[:,i,0],constant_predictions[:,i]))

        for j,prediction in enumerate(predictions):

            time_wise_mae[j].append(mae(truths[:,i,0],prediction[:,i,0]))
        
    for i in range (len(predictions)):
        plt.scatter(time_from_last_x, time_wise_mae[i], color = colors[i],\
             marker='x', label = 'cols [{}]'.format(labels[i]))

    plt.scatter(time_from_last_x, time_wise_mae_constant, color = 'black', marker='x', label = 'Constant')
    plt.xlabel('Time from last seen value by SCINet')
    plt.ylabel('Mean Absolute Error')
    plt.grid()
    plt.legend()
    plt.title('Mean absolute error from last seen value')


plt.show()