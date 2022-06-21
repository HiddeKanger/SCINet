import tensorflow as tf
import numpy as np

# Splitting layer:
splitting_layer = tf.keras.layers.Lambda(lambda x: (x[:, ::2, :],
                                                    x[:, 1::2, :]))

# Class for the replication padding:
class ReplicationPadding1D(tf.keras.layers.Layer):
    '''
    This is a recreation of the ReplicationPad1d class from Torch.
    It basically copies the border of the input given an amount of
     padding at the begining and at the end.
    '''
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding1D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[1] + self.padding[0] + self.padding[1]

    def call(self, input_tensor):
        padding_left, padding_right = self.padding
        return tf.pad(input_tensor,  [[0, 0], [padding_left, padding_right],
         [0, 0]], mode='SYMMETRIC')


class InteractorLayer(tf.keras.layers.Layer):
    '''
    This class is a custom layer for the SCI-block. The Interactor class
    in the Torch implementation.
    All 4 "modules" have the same structure, as defined here, so they will be
     treated as layers.
    There is no transposition because the current shape (BATCH-TIME-DIMENSIONS)
     is coherent with tensorflow expected shapes.
    
    Here potential combinations of different channels are taken into account
     due to the convolutional layers over the time dimension.
    '''
    def __init__(self, in_planes, kernel = 5, dropout=0.5, hidden_size = 32,
     **kwargs):
        super().__init__(**kwargs)

        self.kernel_size = kernel
        self.dilation = 1

        if self.kernel_size % 2 == 0: # Just to make the series digestible by the conv1D with "constant" padding
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1 #by default: stride==1 
            pad_r = self.dilation * (self.kernel_size) // 2 + 1 #by default: stride==1 

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1

        self.rep_padding = ReplicationPadding1D(padding=(pad_l, pad_r))

        self.conv1 = tf.keras.layers.Conv1D(
                filters = int(in_planes * hidden_size), # In planes must then be the number of dimensions.
                kernel_size = kernel ,
                dilation_rate= 1 , # Fixed by them
                strides= 1, # Fixed by them
            )
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.01)
        self.drop = tf.keras.layers.Dropout(rate = dropout)
        self.conv2 = tf.keras.layers.Conv1D(
                filters= in_planes,
                kernel_size= 3, 
                strides = 1,
                activation= "tanh")

    
    def call(self, x, training = True):
        Z = x
        Z = self.rep_padding(Z)
        Z = self.conv1(Z)
        Z = self.leaky_relu(Z)
        if training:
            Z = self.drop(Z)
        Z = self.conv2(Z)
        return(Z)


class Sci_block(tf.keras.layers.Layer):
    '''
    This layer performs the interaction between the 4 modules.
    '''
    def __init__(self, in_planes, splitting = True, kernel = 5, dropout=0.5, hidden_size = 1, INN = True, **kwargs):
        super().__init__(**kwargs)
        
        self.splitting = splitting
        self.modified = INN

        # The permutation is performed before entering in the interactorlayer
        self.phi = InteractorLayer(in_planes, kernel= kernel, dropout= dropout, hidden_size= hidden_size)
        self.psi = InteractorLayer(in_planes, kernel= kernel, dropout= dropout, hidden_size= hidden_size)
        self.U = InteractorLayer(in_planes, kernel= kernel, dropout= dropout, hidden_size= hidden_size)
        self.P = InteractorLayer(in_planes, kernel= kernel, dropout= dropout, hidden_size= hidden_size)

    def call(self, x, training = True):
        if self.splitting:
            (x_even, x_odd) = splitting_layer(x)
        else:
            (x_even, x_odd) = x 
        
        if self.modified: 

            d = tf.multiply(x_odd, tf.exp(self.phi(x_even, training)))
            c = tf.multiply(x_even, tf.exp(self.psi(x_odd, training)))

            x_even_update = c + self.U(d, training)
            x_odd_update = d - self.P(c, training)

            return(x_even_update, x_odd_update)

        else:

            d = x_odd - self.P(x_even, training)
            c = x_even + self.U(d, training)

            return (c, d)


class SCINet_Tree(tf.keras.layers.Layer):
    '''
    This layer creates the tree of SCI_block layers. It does so in a recursive way.
    -> current_level: number of trees
    
    They include a further layer just due to this change of name, not necessary.
        
    '''
    def __init__(self, in_planes,
        current_level,
        kernel = 5,
        dropout=0.5,
        hidden_size = 1,
        **kwargs):

        super().__init__(**kwargs)
        self.current_level = current_level
        self.workingblock = Sci_block(
            in_planes = in_planes,
            hidden_size = hidden_size,
            kernel= kernel,
            dropout = dropout) 
        
        if current_level!=0: 
            self.SCINet_Tree_odd=SCINet_Tree(in_planes, kernel= kernel,
              dropout= dropout, 
              hidden_size= hidden_size,
              current_level = current_level-1)

            self.SCINet_Tree_even=SCINet_Tree(in_planes,
                kernel= kernel,
                dropout= dropout, 
                hidden_size= hidden_size, 
                current_level = current_level-1)

    def zip_up_the_pants(self, even, odd):
        '''
        This correctly ensambles the original values after the transformations.
        "After going through L levels of SCI-Blocks, we rearrange the elements in all the sub-features
        by reversing the odd-even splitting operation [...]"
        '''
        even = tf.transpose(even, perm = [1, 0, 2])
        odd = tf.transpose(odd, perm = [1, 0, 2])
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))

        _ = []

        for i in range(mlen):
            _.append(tf.expand_dims(even[i], 0)) # Adds i time entry over ALL dimensions.
            _.append(tf.expand_dims(odd[i], 0))
        if odd_len < even_len:
            _.append(tf.expand_dims(even[-1], 0))
        return tf.transpose(tf.concat(_, 0), perm = [1, 0 ,2]) # Return to original shape.
    
    def call(self, x, training = True): 
        x_even_update, x_odd_update= self.workingblock(x, training)
        if self.current_level ==0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))


class SCINet(tf.keras.layers.Layer):
    '''
    Main model class
    '''
    def __init__(self, output_len, input_len, input_dim, output_dim, hid_size = 1,
                num_levels = 4, kernel = 5, dropout = 0.5, **kwargs):

        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.kernel_size = kernel
        self.dropout = dropout
        self.output_dim = output_dim
        

        self.blocks1 = SCINet_Tree(
            in_planes= self.input_dim,
            current_level = self.num_levels,
            kernel = self.kernel_size,
            dropout = self.dropout,
            hidden_size = self.hidden_size)

        assert self.input_dim % self.output_dim == 0 # Check that the dimension
                                                     # reduction makes sense
        kernel_size = int(self.input_dim / self.output_dim)

      
        
        self.projection1 = tf.keras.layers.Conv1D(self.output_len,
                                                kernel_size = kernel_size,
                                                strides = kernel_size,
                                                use_bias = False)

    def call(self, x, training = True):
        assert self.input_len % (np.power(2, self.num_levels)) == 0
        res1 = x # The original data
        x = self.blocks1(x, training) # To the Tree block
        x += res1
        x = tf.transpose(x, perm = [0, 2, 1]) # Transpose to make the convolutions through all the time steps.
        x = self.projection1(x)
        x = tf.transpose(x, perm = [0, 2, 1]) # Turn again to BATCH-TIME-CHANNELS
        return x


def scinet_builder( output_len: list,
                    input_len: int,
                    output_dim: list,
                    input_dim: int,
                    selected_columns: list, 
                    loss_weights: list,
                    hid_size = 1,
                    num_levels = 4,
                    kernel = 5,
                    dropout = 0.5,
                    learning_rate = 0.01,
                    probabilistic = False # Not used
                ):

    '''
    Returns a compiled SCINet model. 
    
    Arguments:
    ----------
    output_len: list
        Length of the output in each SCINet. If just one SCINet [n] where
        n is the lenght.

    input_len: int
        Length of the input of the *first* SCINet. Subsequent ones are
        determined by previous output length.
        
    output_dim: list
        Dimensionality of the output of each SCINet. If just one SCINet [n]
        where n is the number of dimensions.

    input_dim: int
        Dimensionality of of the *first* SCINet. Subsequent ones are
        determined by previous output dim.
    
    selected_columns: list (or None)
        The index of the columns used to calculate the loss for a particular
        SCINet. When the output_dim != from input_dim the loss of the output
        must be calculated with a subset of the input.
        It must be a list with the lists of columns used to calculate the loss
        on each SCINet. For instance, for 3 SCINet:
            [[1,2,3], [1,2,3], [2]]
    
    loss_weight: list
        Relative importance of the loss for the output of each SCINet.

    hid_size: int
        Number of filters, applicable to all SCINet.

    num_levels: int
        Number of splits, applicable to all SCINet.
    
    kernel: int
        Kernel size, applicable to all SCINet.
    
    dropout: float
        Dropout rate, applicable to all SCINet.

    learning_rate: float
        Learning rate for ADAM.
   
    '''
    # Num examples needed in the case of using probabilistic layer.
    print('Building model...')
    assert len(output_dim) == len(output_len), "output_dim list does not equal length output_len"
    assert input_len % 2**num_levels == 0, f"input_len (X_LEN) does not match depth {input_len} % 2**{num_levels} != 0"
    assert (input_len / 2**num_levels) % 2 == 0 , f"input_len (X_LEN) does not match depth. {input_len} / 2**{num_levels} must be even!"

    if selected_columns is not None:
        assert all([output_dim[i] == len(selected_columns[i])
            for i in range(len(output_dim))]), 'Output_dims and selected columns do not correspond' #Making sure inputs are coherent

    
    inputs = tf.keras.Input(shape = (input_len, input_dim))

    X = inputs
    outputs = []
    # Loop over the number of output dimensions provided, if just one [x]
    # is provided, it has just one stack.
    input_dims = [input_dim] + output_dim


    for i in range(len(output_dim)):
        
        assert X.shape[1] == input_len # Sanity check

        x_mid = SCINet(output_len= output_len[i],
                    input_len= input_len,
                    input_dim = input_dims[i],
                    num_levels = num_levels, 
                    kernel= kernel,
                    hid_size = hid_size,
                    output_dim= output_dim[i],
                    dropout= dropout,
                    name = f'Block_{i}')(X)

        outputs.append(x_mid)
        if i == (len(output_dim) - 1):
            break
        
        if selected_columns is not None: # If input dim != output dime
            int_inputs = tf.gather(inputs, selected_columns[i], axis = 2)
        else:
            int_inputs = inputs
            
        new_input = tf.concat([int_inputs, x_mid], 1)
        X = tf.keras.layers.Cropping1D((output_len[i], 0))(new_input)
        # Removing "old" prices first
    

    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate),
         loss = {f'Block_{i}':"mae" for i in range(len(output_dim))},
            loss_weights = loss_weights)


    return model


if __name__ == "__main__":
    ## EXAMPLE TO RUN
    X = np.random.rand(1, 800, 5)

    my_layer = scinet_builder(output_len= [20]*5, input_len= 800, input_dim= 5,
    output_dim= [5]*5, num_levels = 2, loss_weights= [0.2]*5, selected_columns=None)
    print('Model compiled')
    print(my_layer.summary())
    print(my_layer.losses)
    # tf.keras.utils.plot_model(my_layer,
    #  to_file= "./model.png")

    print(my_layer(X))