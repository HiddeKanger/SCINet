# Reproduction of Paper Results


The experiments in this folder (reprod) are set up in order to reproduce and compare the results from the original paper.
It uses the datasets ETTh1, ETTh2, ETTm1 and exchange_rate. In each experiment, the dataset is first preprocessed, and afterwards
the SCINet is trained on the data. Next, the results of the MAE loss are compared against the MAE value of the original paper,
and the predictions made by SCINet based on the given dataset are plotted as well.


# Table of contents

- data (folder):
    - Data_original (folder): datasets used by the original paper
    - Data_preprocessed (folder): datasets used by the experiments, preprocessed such that they are compatible for the experiments
- results (folder): contains the results of the loss and the predictions produced by the experiments
- experiment_ETTh1.py: experiment using the ETTh1.csv dataset
- experiment_ETTh2.py: experiment using the ETTh2.csv dataset
- experiment_ETTm1.py: experiment using the ETTm1.csv dataset
- experiment_exchange_rate.py: experiment using the exchange_rate.csv dataset

The files experiment_ETTh1.py, experiment_ETTh2.py, experiment_ETTm1.py use the hyperparameter settings corresponding to the horizon
of 24 (Y_LEN=24), and experiment_exchange_rate.py the hyperparameter settings corresponding to the horizon of 12 (Y_LEN=12).

The hyperparameter settings can be found at https://github.com/cure-lab/SCINet/blob/main/Appendix/Appendix.pdf

# Running the experiments

In order to run the experiments, make sure to use an environment in which Tensorflow 2.x is installed. Running can be done by executing 
"python experiment_X.py" from the command line. 