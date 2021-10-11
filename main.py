print("STARTING ECG ANALYSIS PROGRAM")

import os
import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow.errors import InvalidArgumentError
from tensorflow import keras
import pandas as pd
import matplotlib as plt
from ECGNet import ECGnet
from genFunc import generationFunctions
from printFunc import PrintFunctions

print("FINISHED IMPORTS")

ECGFOLDER = 'ECGDataDenoised'
DIAGNOSTICS = pd.read_excel('Diagnostics.xlsx')

print("LOADED ECG FILES")

#Function that grabs largest value from returned logits for a batch
def get_predictions(logit_tensor):
    prediction_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    logit_tensor_shape = tf.shape(logit_tensor)[0]
    for i in range(logit_tensor_shape):
        #Write largest value of each logit set to a tensorarray
        prediction_array = prediction_array.write(i, tf.cast(tf.argmax(logit_tensor[i, 0]), tf.int32))
    #convert to a tensor before returning
    prediction_tensor = prediction_array.stack()
    return prediction_tensor

#Run a batch of inputs at a certain index and edit the gradients if training
def run_batch(index, batchsize, ecgmodel, y_true, ecglist, opt, training = True):
    #Initialize a gradient tape to watch our tensors
    with tf.GradientTape() as tape:
        #Initialize empty tensorarrays
        y_true_batch = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        y_pred_batch = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        indexwrite = 0
        #Main loop
        for i in range(index, (index + batchsize), 1):
            if (index + batchsize) > (len(ecglist) - 1):
                break
            #Read file at index
            inputval = pd.read_csv(os.getcwd() + '/' + ECGFOLDER + '/'+ ecglist[i])
            #Sometimes Nan values get in the input value, just replace them with 0s
            inputval.fillna('0', inplace = True)
            inputval = inputval.values
            inputval = inputval.astype(np.float64)
            #Force the tensor to be converted to a float64 tensor to ensure the nodes values are correct
            inputval = tf.Variable(inputval)
            inputval = tf.cast(inputval, tf.float64)
            #Check for invalid sets
            try:
                inputval = tf.reshape(inputval, [1, 4999, 12, 1])
            except:
                #Print the valid value so we can remove it from the set
                print(ecglist[i])
            #write our tensorArrays with the true values and predicted values
            y_true_batch = y_true_batch.write(indexwrite, y_true[i])
            y_pred_batch = y_pred_batch.write(indexwrite, ecgmodel(inputval))
            indexwrite += 1
        #Stack the tensorArrays for the ahead functions
        y_true_batch = y_true_batch.stack()
        y_pred_batch = y_pred_batch.stack()
        #y predict batch is the list of values that the net is predicting, while the pred object is the
        #activations at the last layer
        y_predict_batch = get_predictions(y_pred_batch)
        #Calculating loss with sparse categorical entropy because we are using integer labels
        scce = keras.losses.SparseCategoricalCrossentropy()
        #Simple accuracy calculation, 1 for correct answer, 0 for incorrect
        acc = keras.metrics.Accuracy()
        loss = scce(y_true_batch, y_pred_batch)
        #Update the accuracy
        acc.update_state(y_true_batch, y_predict_batch)
        #Read the accuracy
        accuracy = acc.result().numpy()
    #Only if were training do we apply the gradients, if we do such on testing data the model will
    #memorize/overfit on the test data
    if training:
        gradients = tape.gradient(loss, ecgmodel.trainable_variables)
        opt.apply_gradients(zip(gradients, ecgmodel.trainable_variables))
    return accuracy, loss

#Epoch is a collection of batchs such that every sample within the data is used
def run_epoch(ecgmodel, opt, epoch_num, ecglist, batchsize = 32, shuffle = True, training = True):
    if training:
        print("Running Epoch: " + str(epoch_num))
    else:
        print("TESTING")
    if shuffle:
        random.shuffle(ecglist)
    #index is the ecg file we're on
    index = 0
    #get the expected values from our data
    y_true = gen_y_true(ecglist)
    #counter is the batch num we're on
    counter = 0
    #tensorArrays to store accuracy and loss of each batch for Epoch
    accuracyarray = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    lossarray = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    while True:
        try:
            accuracy, loss = run_batch(index, batchsize, ecgmodel, y_true, ecglist, opt, training = training)
        except ValueError:
            #we hit a value error when we don't have enough left to run a proper batch,
            #however becuase we shuffle for every epoch we ensure every sample is hit
            print("FINISHED EPOCH")
            break
        index += batchsize
        accuracyarray = accuracyarray.write(counter, accuracy)
        lossarray = lossarray.write(counter, loss)
        avgaccuracy = tf.reduce_mean(accuracyarray.stack())
        avgloss = tf.reduce_mean(lossarray.stack())
        #show the progreess in the terminal
        printProgressBar(counter, int(len(ecglist)/batchsize), prefix = "batch: " + str(counter) + "/" + str(int(len(ecglist)/batchsize)), suffix = 'Accuracy:' + str(avgaccuracy.numpy()) + ' ' + 'Loss: ' + str(avgloss.numpy()), decimals = 1, length = 40)
        counter += 1

#Main function that takes the model, optimizer, and number of Epochs
def train_model(model, opt, num=10):
    ecglist = os.listdir(ECGFOLDER)
    ecglist, testlist = gen_test_list(ecglist)
    for i in range(num):
        #training the model
        run_epoch(model, opt, i, ecglist)
        run_epoch(model, opt, i, testlist, training=False)
#main
ECGMODEL = ECGnet()
opt = keras.optimizers.Adam(learning_rate = 0.0001)
ECGMODEL.summary()
train_model(ECGMODEL, opt)

    
    
            
            


