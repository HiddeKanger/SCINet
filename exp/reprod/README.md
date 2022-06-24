# Reproduction of Paper Results


This folder contains the files used for reproducing the results found by the SCINet model from the original paper. The datasets used by the orignal paper are used in order to evaluate the performance of our model. The original hyperparameter settings used by the paper are used here.


Files:

- `data/`: contains the folders `Data_original/` and `Data_preprocessed/` which contain the datasets used by the orignal paper, and those datasets reshaped for our model respectively,
- `saved_models/`: contains all the saved weights of the trained models
- `utils/`: several supporting files used by the reproduction of the results
- `results/`: results, plots outputted by the scripts in this folder
- `reprod_experiments.ipynb`: Evaluates the difference between the leaky Tensorflow implementation and and the non-leaky variant in order to compare the difference in the results between the techniques used by the orignal paper and our implementation, as well as evaluate the model's performance.