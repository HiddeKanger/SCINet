import numpy as np
import matplotlib.pyplot as plt

def plot_raw_data(raw_data, x_lens):

    time = raw_data[:,0]
    n_samples = raw_data.shape[0]
    dt = raw_data[1,0]-raw_data[0,0]

    colors = ['blue', 'red', 'green']
    labels = ['Feature I','Feature II','Feature III']
    xlabels = ['Time in (a.u.)','Time in (a.u.)','Time in (a.u.)','Period in (a.u.)']
    ylabels = 'Amplitude in (a.u.)'
    subtitels = ['First {} timesteps of the different features'.format(x_lens[0]),
             'First {} timesteps of the different features'.format(x_lens[1]),
             'First {} timesteps of the different features'.format(x_lens[2]),
             'Fourier transforms of the different time series'
             ]

    fig, axes = plt.subplots(1,4, figsize = (20,4))

    for i in range(1,raw_data.shape[1]):

        axes[0].plot(time[:x_lens[0]],raw_data[:x_lens[0],i], c = colors[i-1], label = labels[i-1])
        axes[1].plot(time[:x_lens[1]],raw_data[:x_lens[1],i], c = colors[i-1], label = labels[i-1])
        axes[2].plot(time[:x_lens[2]],raw_data[:x_lens[2],i], c = colors[i-1], label = labels[i-1])

        fft = np.fft.fft(raw_data[:,i])
        fftfreq = np.fft.fftfreq(n_samples, dt)

        axes[3].semilogx(1/(fftfreq+1e-9), np.abs(fft), c = colors[i-1], label = labels[i-1])

    for i in range(4):

        axes[i].grid()
        axes[i].legend(ncol = 3, loc = 'upper center')
        axes[i].set_xlabel(xlabels[i])
        axes[i].set_ylabel(ylabels)
        axes[i].set_title(subtitels[i])

    axes[3].set_xlim(1,1200)

    fig.suptitle('Visualization of the raw data at different timescales', fontsize = 20)
    plt.tight_layout()

def plot_preprocessed_data(X,y):

    fig, ax = plt.subplots(1,1, figsize = (8,4))

    x_time = np.arange(X.shape[0])
    y_time = np.arange(X.shape[0],X.shape[0]+y.shape[0])
    colors = ['blue', 'red', 'green']
    labels = ['$X_{1}$','$X_{2}$','$X_{3}$','$y_{1}$','$y_{2}$','$y_{3}$']

    for i in range(X.shape[1]):

        ax.plot(x_time, X[:,i], c = colors[i], label = labels[i])
        ax.plot(y_time, y[:,i], c = colors[i], label = labels[i+3], linestyle = 'dashed')

    ax.set_xlabel('Time in (a.u.)')
    ax.set_ylabel('Amplitude in (a.u.)')
    ax.grid()
    ax.legend(ncol = 6, loc = 'upper center', fontsize = 13)

def plot_loss_curves(n_epochs, losses):

    labels = ['Scinet 1 training loss', 'Scinet 2 training loss',\
              'Scinet 1 validation loss',' Scinet 2 validation loss']
    colors = ['black', 'red', 'black', 'red']
    linestyles = ['dashed','dashed','solid','solid']

    fig, ax = plt.subplots(1,1, figsize = (6,4))

    for i,loss in enumerate(losses):
        
        ax.plot(np.arange(n_epochs),loss, c = colors[i], linestyle = linestyles[i], label = labels[i])

    ax.grid()
    ax.legend(ncol =2, loc = 'upper center')
    ax.set_xlabel('Epoch number')
    ax.set_ylabel('Mean absolute error')

    fig.suptitle('Training and validation loss during training', fontsize = 16)
    plt.tight_layout()

def plot_prediction_examples(n_samples, x_samples,y_predictions, y_true):

    fig, axes = plt.subplots(y_true.shape[2], n_samples, figsize=(20,6))

    labels = ['SCINet input', 'Ground truths','SCINet predictions']

    x_val_x = np.arange(x_samples.shape[1])
    x_val_y = np.arange(x_samples.shape[1],x_samples.shape[1]+y_predictions.shape[1])
    x_tot = len(x_val_x)+ len(x_val_y)

    for i in range(y_true.shape[2]):
        for j in range(n_samples):
            line_x, = axes[i,j].plot(x_val_x, x_samples[j,:,i], c = 'black', label = labels[0])
            line_y_true, = axes[i,j].plot(x_val_y, y_true[j,:,i], c = 'black', linestyle = 'dashed', label = labels[1])
            line_y_pred, = axes[i,j].plot(x_val_y, y_predictions[j,:,i], c = 'red', label = labels[2])
            axes[i,0].set_ylabel('Feature {}'.format(i+1), rotation = 90, fontsize = 16)
            axes[i,j].set_xticks(np.arange(0,x_tot+1,10))
            axes[i,j].set_yticks([])
            axes[2,j].set_xlabel('Time in (a.u.)')

    fig.suptitle('Predictions on the features made by SCINet', fontsize = 20)
    plt.legend(handles = [line_x,line_y_true,line_y_pred],  loc = 0, bbox_to_anchor= (-2.7,3.72,0,0), ncol = 3, fontsize = 14)
    plt.tight_layout()
