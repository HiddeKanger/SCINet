# SCINet basics:
Scinet is an alternative to LSTM, TCN or Transformers for (multidimensional) time series forecasting. It is based on splitting the input into even and odd time-step (based on time-step position in the input sequence) recursively in order to achieve an arbitrarily broad receptive field, constructing a representation and concatenating it in the original order to then make predictions based on a linear transformation of this representation. It allows for an arbitrary number of predicted steps.

Scinet is composet of "stacks". Each stack predicts an arbitrary number of steps beyond and this is evaluated with L1 loss. The total loss of the model is the (weighted) sum of the intermediate losses and the final loss. Each stack is feeded the last part of the original input and the predicted steps of the previous stack (mantaining dinput dimensionality) therefore creating a (potentially) more elaborated representation each time.

The input is structured as batch-time-dimensions. 

The outputs, however, do not need to have the same dimensionality as the inputs (this is not present on the original implementation). For instance, it is possible to feed an arbitrary number of stocks and/or indicators (such as volume, maximum price...) and output just the close price for one particular stock and, therefore, focusing the loss on it. 

# SCINet.py usage:

Importing "scinet_builder" allows a lot of flexibility to construct all types of models. The general idea is that the number of "stacks" is defined by the number of entries in the lists that the function take as arguments. 

For instance, if we want to use volume and close price for 3 different stocks but we are only interested in one stock, we can structure the model such that the first stack outputs just the close price of the 3 stocks and second stack just outputs the close price of the stock we are interested in. In order to do so we call the "scinet_builder" and specify the output dimension ("output_dim") of each stack as [[3], [1]] and then select the index of the columns ("selected_columns") in the input that we want to use to calculate the loss in each stack as [[x, y, z], [z]]. 

The relative importance of the loss can be balanced through the "loss_weight" argumen
