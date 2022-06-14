import numpy as np
import matplotlib.pyplot as plt

# General settings
N_timesteps = 50000
N_features = 3
time = np.arange(N_timesteps)

# Settings Feature I
period1 = 1000
amplitude1 = 1
noise1 = 0.1

# Settings Feature II
period2 = 10
amplitude2 = 0.5
noise2 = 0.1

# Settings Feature III
amplitude3 = 1

# Make dataset

data = np.zeros((N_timesteps,N_features+1))

data[:,0] = time
data[:,1] = amplitude1 * np.sin(2*np.pi/period1*time)+np.random.normal(loc=0,scale=noise1,size = len(time))
data[:,2] = amplitude2 * np.sin(2*np.pi/period2*time)+np.random.normal(loc=0,scale=noise2,size = len(time))
data[:,3] = amplitude3 * np.sin(data[:,1]*data[:,2])

np.savetxt('datasets/toy_dataset_sine.csv', data)

if __name__ == '__main__':

    window_length = 100

    fig, (ax1,ax2,ax3) = plt.subplots(1,3)

    ax1.plot(data[:window_length,0] , data[:window_length,1])
    ax2.plot(data[:window_length,0] , data[:window_length,2])
    ax3.plot(data[:window_length,0] , data[:window_length,3])

    plt.show()

