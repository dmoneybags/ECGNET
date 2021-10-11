#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class ECGnet(keras.Sequential):
    def __init__(self, _input_shape = [4999, 12, 1], _num_conditions = 11, _kernel_size = (10,10), _dilations = 10, _num_hidden_units = 128):
        print("INITIALIZING ACTOR")
        super(ECGnet, self).__init__([
            #Normalize the data to reduce sensitivity to individual values and increase training stability
            keras.layers.BatchNormalization(name = "Normalization", input_shape = _input_shape),
            #Initialize our Conv2d layer with a kernel of 10, 10 and 10 dilations to generate activations for next layer
            keras.layers.Conv2D(_dilations, _kernel_size, activation = 'relu', padding = "same", name = "FirstConv2d"),
            #Begin our sequence of Alternating Conv2d Layers and Max pooling layers to prevent overfitting to various locations within the Data
            keras.layers.Conv2D(_dilations * 2, _kernel_size, activation = 'relu', padding = "same", name = "SecConv2d"),
            keras.layers.MaxPooling2D(padding = "same", name = "FirstMaxPooling"),
            keras.layers.Conv2D(_dilations * 2, _kernel_size, activation = 'relu', padding = "same", name = "ThirdConv2d"),
            keras.layers.MaxPooling2D(padding = "same", name = "SecMaxPooling"),
            #Dropout to prevent overfitting
            keras.layers.Dropout(0.2, name = "Dropout"),
            keras.layers.Conv2D(_dilations * 2, _kernel_size, activation = 'relu', padding = "same", name = "FourthConv2d"),
            #Flattening to pass the activations to Dense Layer
            keras.layers.Flatten(name = "Flatten"),
            #Dense Layer to increase depth and node count
            keras.layers.Dense(_num_hidden_units, activation = 'relu', name = "Dense"),
            #Softmax output layer
            keras.layers.Dense(_num_conditions, activation = 'softmax')
        ])
    def call(self, x, training = True):
        #pass input through every Layer
        for layer in self.layers:
            x = layer(x)
        #return last set of activations from softmax layer
        return x

