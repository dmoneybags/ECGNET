#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow.errors import InvalidArgumentError
from tensorflow import keras

ECGFOLDER = 'ECGDataDenoised'
DIAGNOSTICS = pd.read_excel('Diagnostics.xlsx')

class generationFunctions:
    
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

