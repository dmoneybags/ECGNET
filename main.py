print("STARTING ECG ANALYSIS PROGRAM")

import os
import sys
import numpy as np
import random
from selenium.common.exceptions import InvalidArgumentException
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
            #keras.layers.Reshape((4999, 12, 1), input_shape = [4999, 12, 1]),
            keras.layers.BatchNormalization(name = "Normalization", input_shape = _input_shape),
            keras.layers.Conv2D(_dilations, _kernel_size, activation = 'relu', padding = "same", name = "FirstConv2d"),
            keras.layers.Conv2D(_dilations * 2, _kernel_size, activation = 'relu', padding = "same", name = "SecConv2d"),
            keras.layers.MaxPooling2D(padding = "same", name = "FirstMaxPooling"),
            keras.layers.Conv2D(_dilations * 2, _kernel_size, activation = 'relu', padding = "same", name = "ThirdConv2d"),
            keras.layers.MaxPooling2D(padding = "same", name = "SecMaxPooling"),
            keras.layers.Dropout(0.2, name = "Dropout"),
            keras.layers.Conv2D(_dilations * 2, _kernel_size, activation = 'relu', padding = "same", name = "FourthConv2d"),
            #keras.layers.MaxPooling2D(padding = "same", name = "ThirdMaxPooling"),
            keras.layers.Flatten(name = "Flatten"),
            keras.layers.Dense(_num_hidden_units, activation = 'relu', name = "Dense"),
            #keras.layers.Dense(_num_hidden_units/2, activation = 'relu', name = "Dense2"),
            keras.layers.Dense(_num_conditions, activation = 'softmax')
        ])
    def call(self, x, training = True):
        for layer in self.layers:
            x = layer(x)
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

def print_ecg_data():
    ecgfile = os.listdir(ECGFOLDER)[0]
    ecgdata = pd.read_csv(os.getcwd() + '/' + ECGFOLDER + '/'+ ecgfile,sep=',',header=None)
    print(tf.shape(ecgdata.values))

def gen_rhythm_dict():
    rhythm_dict = {}
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

def gen_test_list(ecglist, num_subjects=500):
    if ".DS_Store" in ecglist:
        print("CLEANING ECGLIST")
        ecglist.remove(".DS_Store")
    num_ecg = len(ecglist)
    print("NUMBER OF SUBJECTS IS: " + str(num_ecg))
    indices = []
    testlist = []
    for i in range(num_subjects):
        while True:
            indice = random.randint(0, num_ecg - 1)
            if indice not in indices:
                indices.append(indice)
                break
    for indice in indices:
        testlist.append(ecglist[indice])
    for val in testlist:
        ecglist.remove(val)
    return ecglist, testlist

def gen_y_true(ecglist):
    y_true = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rhythm_dict = gen_rhythm_dict()
    index = 0
    for filename in ecglist:
        datarow = DIAGNOSTICS.loc[DIAGNOSTICS['FileName'] == filename[:-4]]
        rhythm = datarow['Rhythm']
        y_true = y_true.write(index, rhythm_dict[rhythm.values[0]])
        index += 1
    print("GENERATED LABELS")
    return y_true.stack()

def get_predictions(logit_tensor):
    prediction_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    logit_tensor_shape = tf.shape(logit_tensor)[0]
    for i in range(logit_tensor_shape):
        prediction_array = prediction_array.write(i, tf.cast(tf.argmax(logit_tensor[i, 0]), tf.int32))
    prediction_tensor = prediction_array.stack()
    return prediction_tensor

def run_batch(index, batchsize, ecgmodel, y_true, ecglist, opt, training = True):
    with tf.GradientTape() as tape:
        y_true_batch = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        y_pred_batch = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        indexwrite = 0
        for i in range(index, (index + batchsize), 1):
            if (index + batchsize) > (len(ecglist) - 1):
                break
            inputval = pd.read_csv(os.getcwd() + '/' + ECGFOLDER + '/'+ ecglist[i])
            inputval.fillna('0', inplace = True)
            inputval = inputval.values
            inputval = inputval.astype(np.float64)
            inputval = tf.Variable(inputval)
            inputval = tf.cast(inputval, tf.float64)
            try:
                inputval = tf.reshape(inputval, [1, 4999, 12, 1])
            except:
                print(ecglist[i])
            y_true_batch = y_true_batch.write(indexwrite, y_true[i])
            y_pred_batch = y_pred_batch.write(indexwrite, ecgmodel(inputval))
            indexwrite += 1
        y_true_batch = y_true_batch.stack()
        y_pred_batch = y_pred_batch.stack()
        y_predict_batch = get_predictions(y_pred_batch)
        scce = keras.losses.SparseCategoricalCrossentropy()
        acc = keras.metrics.Accuracy()
        loss = scce(y_true_batch, y_pred_batch)
        acc.update_state(y_true_batch, y_predict_batch)
        accuracy = acc.result().numpy()
    if training:
        gradients = tape.gradient(loss, ecgmodel.trainable_variables)
        opt.apply_gradients(zip(gradients, ecgmodel.trainable_variables))
    return accuracy, loss

def run_epoch(ecgmodel, opt, epoch_num, ecglist, batchsize = 32, shuffle = True, training = True):
    if training:
        print("Running Epoch: " + str(epoch_num))
    else:
        print("TESTING")
    if shuffle:
        random.shuffle(ecglist)
    index = 0
    y_true = gen_y_true(ecglist)
    counter = 0
    accuracyarray = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    lossarray = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    while True:
        try:
            accuracy, loss = run_batch(index, batchsize, ecgmodel, y_true, ecglist, opt, training = training)
        except ValueError:
            print("FINISHED EPOCH")
            break
        index += batchsize
        print(index)
        accuracyarray = accuracyarray.write(counter, accuracy)
        lossarray = lossarray.write(counter, loss)
        avgaccuracy = tf.reduce_mean(accuracyarray.stack())
        avgloss = tf.reduce_mean(lossarray.stack())
        printProgressBar(counter, int(len(ecglist)/batchsize), prefix = "batch: " + str(counter) + "/" + str(int(len(ecglist)/batchsize)), suffix = 'Accuracy:' + str(avgaccuracy.numpy()) + ' ' + 'Loss: ' + str(avgloss.numpy()), decimals = 1, length = 40)
        counter += 1

def train_model(model, opt, num=10):
    ecglist = os.listdir(ECGFOLDER)
    ecglist, testlist = gen_test_list(ecglist)
    for i in range(num):
        run_epoch(model, opt, i, ecglist)
        run_epoch(model, opt, i, testlist, training=False)
#main
#WE BEAT AVG DOC
#0.827 acc to beat with 10 dilations giving us the push
ECGMODEL = ECGnet()
opt = keras.optimizers.Adam(learning_rate = 0.0001)
ECGMODEL.summary()
train_model(ECGMODEL, opt)

    
    
            
            


