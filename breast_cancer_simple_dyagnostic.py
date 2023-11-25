import pandas as pd
import numpy as np

# Slip data library
from sklearn.model_selection import train_test_split

# Keras Library
import keras
from keras.models import Sequential
from keras.layers import Dense

# Load data
predictors = pd.read_csv('entradas_breast.csv')
classification = pd.read_csv('saidas_breast.csv')

# Train and Test Data Split
pre_train, pre_test, class_train, class_test = train_test_split(predictors, classification, test_size=0.25)

classifier = Sequential()

# Simple equation to set the start number of hidden neurons
# (n_inputs + n_outputs)/2
n_inputs = 30
n_outputs = 1
num_neurons = round((n_inputs + n_outputs)/2)

# units is the number of neurons used in the hidden layer 
# activation is the activation function
# kernel_initializer is how the weights will be initialized
# The input_dim set the input dimention of the hidden layers, therefore the input dimention is implicit
classifier.add(Dense(units = num_neurons, activation = 'relu',
                    kernel_initializer= 'random_uniform', input_dim=30))

# For the output, the sigmoid is used since is a binary classification ploblem
classifier.add(Dense(units=1, activation='sigmoid'))














