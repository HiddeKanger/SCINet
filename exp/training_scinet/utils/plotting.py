import numpy as np
import matplotlib.pyplot as plt

def plot_raw_data(raw_data, ax1_x_len, ax2_x_len):

    time = raw_data[:,0]
    n_samples = raw_data.shape[0]
    dt = raw_data[1,0]-raw_data[0,0]

    colors = ['blue', 'red', 'green']
    labels = ['Feature I','Feature II','Feature III']
    xlabels = ['Time in (a.u.)','Time in (a.u.)','Time in (a.u.)','Period in (a.u.)']
    ylabels = 'Amplitude in (a.u.)'
    subtitels = ['First {} timesteps of the different features'.format(ax1_x_len),
             'First {} timesteps of the different features'.format(ax2_x_len),
             'All timesteps of the different features',
             'Fourier transforms of the different time series'
             ]

    fig, axes = plt.subplots(1,4, figsize = (20,4))

    for i in range(1,raw_data.shape[1]):

        axes[0].plot(time[:ax1_x_len],raw_data[:ax1_x_len,i], c = colors[i-1], label = labels[i-1])
        axes[1].plot(time[:ax2_x_len],raw_data[:ax2_x_len,i], c = colors[i-1], label = labels[i-1])
        axes[2].plot(time,raw_data[:,i], c = colors[i-1], label = labels[i-1])

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