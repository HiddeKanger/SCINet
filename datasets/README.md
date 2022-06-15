# Datasets folder

This folder contains toy datasets that are useful in benchmarking and illustrating SCINet. It includes the following sets:

**toy_dataset_sine.csv**

Very simple dataset that can be used to verify that SCINet is learning at all. It consists of three features (and a first column for the timestamps) of length 10000 and dt = 1. These features are generated as such:
- Feature 1: sin(2*pi/1000 * t) + N(0,0.1)
- Feature 2: 0,5 * sin(2*pi/100 * t) + N(0,0.1)
- Feature 3: Feature1 * Feature 2

Here, 'N' denotes Gaussian random noise with its first argument the bias and its second the standard deviation.
