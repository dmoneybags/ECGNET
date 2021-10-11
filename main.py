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

print("FINISHED IMPORTS")

ECGFOLDER = 'ECGDataDenoised'
DIAGNOSTICS = pd.read_excel('Diagnostics.xlsx')

print("LOADED ECG FILES")

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
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = " "):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print(o)
        print()

#Unused function to check ECG data
def print_ecg_data():
    ecgfile = os.listdir(ECGFOLDER)[0]
    ecgdata = pd.read_csv(os.getcwd() + '/' + ECGFOLDER + '/'+ ecgfile,sep=',',header=None)
    print(tf.shape(ecgdata.values))

#Generating our Rythmn dictionary which we'll use to create an integer representation for our condition
def gen_rhythm_dict():
    rhythm_dict = {}
    #We use a dictionary because hashing allows for quicker access than finding within a List
    rhythm_dict['SB'] = 0
    rhythm_dict['SR'] = 1
    rhythm_dict['AFIB'] = 2
    rhythm_dict['ST'] = 3
    rhythm_dict['AF'] = 4
    rhythm_dict['SA'] = 5
    rhythm_dict['SVT'] = 6
    rhythm_dict['AT'] = 7
    rhythm_dict['AVNRT'] = 8
    rhythm_dict['AVRT'] = 9
    rhythm_dict['SAAWR'] = 10
    return rhythm_dict

#Generating a list of Test subjects from our total training data
def gen_test_list(ecglist, num_subjects=500):
    #Remove the auto generated "DS Store" from our ecgList
    if ".DS_Store" in ecglist:
        print("CLEANING ECGLIST")
        ecglist.remove(".DS_Store")
    num_ecg = len(ecglist)
    print("NUMBER OF SUBJECTS IS: " + str(num_ecg))
    indices = []
    testlist = []
    #Pick subjects randomly from the ECGList and add them to the Test List
    for i in range(num_subjects):
        while True:
            indice = random.randint(0, num_ecg - 1)
            if indice not in indices:
                #dont double add one of the same index
                indices.append(indice)
                break
    for indice in indices:
        #add values to our test list
        testlist.append(ecglist[indice])
    for val in testlist:
        #remove them from our original list
        ecglist.remove(val)
    return ecglist, testlist

#Generates a list of the Actual conditions for the ECGList
def gen_y_true(ecglist):
    y_true = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rhythm_dict = gen_rhythm_dict()
    index = 0
    #iterate through files and save their rhythms to a tensorarray
    for filename in ecglist:
        datarow = DIAGNOSTICS.loc[DIAGNOSTICS['FileName'] == filename[:-4]]
        rhythm = datarow['Rhythm']
        y_true = y_true.write(index, rhythm_dict[rhythm.values[0]])
        index += 1
    print("GENERATED LABELS")
    return y_true.stack()

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
#WE BEAT AVG DOC
#0.827 acc to beat with 10 dilations giving us the push
ECGMODEL = ECGnet()
opt = keras.optimizers.Adam(learning_rate = 0.0001)
ECGMODEL.summary()
train_model(ECGMODEL, opt)

    
    
            
            


