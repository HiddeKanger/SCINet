# SCINet basics:
Scinet is an alternative to LSTM, TCN or Transformers for (multidimensional) time series forecasting. It is based on splitting the input into even and odd time-step (based on time-step position in the input sequence) recursively in order to achieve an arbitrarily broad receptive field, constructing a representation and concatenating it in the original order to then make predictions based on a linear transformation of this representation. It allows for an arbitrary number of predicted steps.

Scinet is composet of "stacks". Each stack predicts an arbitrary number of steps beyond and this is evaluated with L1 loss. The total loss of the model is the (weighted) sum of the intermediate losses and the final loss. Each stack is feeded the last part of the original input and the predicted steps of the previous stack (mantaining dinput dimensionality) therefore creating a (potentially) more elaborated representation each time.

The input is structured as batch-time-dimensions. 

The outputs, however, do not need to have the same dimensionality as the inputs (this is not present on the original implementation). For instance, it is possible to feed an arbitrary number of stocks and/or indicators (such as volume, maximum price...) and output just the close price for one particular stock and, therefore, focusing the loss on it. 

### SCINet.py usage:

Importing "scinet_builder" allows a lot of flexibility to construct all types of models. The general idea is that the number of "stacks" is defined by the number of entries in the lists that the function take as arguments. 

For instance, if we want to use volume and close price for 3 different stocks but we are only interested in one stock, we can structure the model such that the first stack outputs just the close price of the 3 stocks and second stack just outputs the close price of the stock we are interested in. In order to do so we call the "scinet_builder" and specify the output dimension ("output_dim") of each stack as [[3], [1]] and then select the index of the columns ("selected_columns") in the input that we want to use to calculate the loss in each stack as [[x, y, z], [z]]. 

The relative importance of the loss can be balanced through the "loss_weight" argument.

### train_scinet.py usage:

With train_scinet, one can train a SCINet after preprocessing the data. This is done using the 'train_scinet' function that also automatically compiles the model. The train_scinet model takes the following arguments:
- X_train: Data with which to train SCINet
- y_train: Labels associated with X_train
- X_val: Validation training set
- y_val: Validation labels
- X_test: Test set data
- y_test: Test set labels
- epochs: Number of epochs with which to train SCINet
- batch_size: Number of samples per training batch
- X_len: Number of timesteps in the input 
- Y_len: List of number of predictions per SCINet in stacked-SCINet
- output_dim: List of number of columns (features) per SCINet in stacked-SCINet
- selected_columns: List of list of columns (relative to original input) to predict per SCINet in stacked-SCINet or 'None'
- hid_size: Number of convolutional filters in SCI-block
- num_levels: Number of layers in SCI-tree
- kernel: Kernel size of convolutional filters in SCI-block
- dropout: Dropout between convolutional layers in SCI-block
- loss_weights: List of weights attributed to loss of different SCINets in stacked-SCINet
- learning_rate: Learning rate of Adam optimizer

### preprocess_data.py usage:

With preprocess_data.py long time series can be prepared for training SCINet. Its principal function is preprocess which is also able to combine multiple data files. Here, the merging happends based on the timestamp column of a pandas dataframe. If a timestamp is not present anywhere it is discarded. The preprocess function takes the following arguments:
- data: Dictionary containing the different datasets to combine
- symbols: List of the names of the different columns in the datasets 
- data_format: keys of the dictionary in data
- fraction: Fraction of the total data to use
- train_frac: Fraction of the data that is used for the train set
- val_frac: Fraction of the data that is used for the validation set
- test_frac: Fraction of the data that is used for the test set
- X_LEN: Number of timesteps per sample used for prediction
- Y_LEN: Number of timesteps per sample to predict
- OVERLAPPING: Whether to make the different samples overlapping (in the time dimension)
- STANDARDIZE: Whether to standardize the data
- standardization_settings: How to standardize the data see 'training_scinet' for an example

