import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as mae



def plot_barplot(type, values, maes, ):

    plt.bar(np.arange(0,len(values)),maes,width = 0.5, align='center', color = 'blue', edgecolor = 'black')
    plt.xticks(np.arange(0,len(values)), values, rotation = 30, fontsize = 14)
    plt.ylabel('Test Mean absolute error')
    plt.xlabel('value for {}'.format(type))
    plt.title(f'Comparison of {type}')
    plt.show()

def plot_prediction_examples(samples, x_samples,y_predictions, y_true):

    fig, axes = plt.subplots(3, len(samples)//3, figsize=(20,6))

    labels = ['SCINet input', 'Ground truths','SCINet predictions']

    x_val_x = np.arange(x_samples.shape[1])
    x_val_y = np.arange(x_samples.shape[1],x_samples.shape[1]+y_predictions.shape[1])
    x_tot = len(x_val_x)+ len(x_val_y)

    for j,sample in enumerate(samples):
        line_x, = axes[j//8,j%8].plot(x_val_x, x_samples[sample,:,0], c = 'black', label = labels[0])
        line_y_true, = axes[j//8,j%8].plot(x_val_y, y_true[sample,:,0], c = 'black', linestyle = 'dashed', label = labels[1])
        line_y_pred, = axes[j//8,j%8].plot(x_val_y, y_predictions[sample,:,0], c = 'red', label = labels[2])
        axes[j//8,0].set_ylabel('Opening price', fontsize = 12)
        axes[j//8,j%8].set_xticks(np.arange(0,x_tot+1,75))
        axes[j//8,j%8].set_yticks([])
        axes[j//8,j%8].set_xlabel('Time in (a.u.)')

    fig.suptitle('Predictions on the opening price made by SCINet', fontsize = 20)
    plt.legend(handles = [line_x,line_y_true,line_y_pred],  loc = 0, bbox_to_anchor= (-2.7,3.72,0,0), ncol = 3, fontsize = 14)
    plt.tight_layout()

def plot_per_timestep_mae(truths,predictions, constant_predictions, Y_len, labels):

    time_from_last_x = np.arange(1,Y_len+1)
    time_wise_mae_constant = []
    time_wise_mae = [ [] for _ in range(len(predictions)) ]

    colors = ['blue']

    for i in range(truths.shape[1]):

        time_wise_mae_constant.append(mae(truths[:,i,0],constant_predictions[:,i]))

        for j,prediction in enumerate(predictions):

            time_wise_mae[j].append(mae(truths[:,i,0],prediction[:,i,0]))
        
    for i in range (len(predictions)):
        plt.scatter(time_from_last_x, time_wise_mae[i], color = colors[i],\
             marker='x', label = labels[i])

    plt.scatter(time_from_last_x, time_wise_mae_constant, color = 'black', marker='x', label = 'Constant')
    plt.xlabel('Time from last seen value by SCINet', fontsize = 12)
    plt.ylabel('Mean Absolute Error', fontsize = 12)
    plt.grid()
    plt.legend()
    plt.title('Mean absolute error at different distances \n from last seen value', fontsize = 14)
    plt.savefig('exp/hyperparams/results/ScinentVSConstant.pdf')



plt.show()