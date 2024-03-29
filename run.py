
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from random import randint
import numpy as np
from NN_class import NN
from LR_class import LR
import os
from sklearn.preprocessing import StandardScaler, normalize


LR_DIMENSIONS = (28*28, 10)
NN_LAYER_DIMENSIONS = (28*28, 10000, 10)
eta = 0.1
epochs = 10
TEST_WITH_LR = False
NORMALIZE_INPUT = True

def main():
    num_classes = 10
    mnist = fetch_openml('mnist_784', parser= 'auto')
    data, target = mnist.data, mnist.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.142857)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    #print(data, target)
    
    #X_train = normalize(X_train, axis=0)
    if NORMALIZE_INPUT:
        print(" --- normalizing input ---")
        # X_train = normalize(X_train)
        # X_test = normalize(X_test)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        #scaler.fit(X_test)
        X_test = scaler.transform(X_test)


    # Loading new dataset
    # X_train, y_train, X_test, y_test = datasetLoading()
  
    Y_train = convert_to_vector(y_train, num_classes)
    Y_test = convert_to_vector(y_test, num_classes)
    
    model = LR(LR_DIMENSIONS[0], LR_DIMENSIONS[1]) if TEST_WITH_LR else NN(NN_LAYER_DIMENSIONS, eta, epochs)

    # For running dataset using batches uncomment the lines below
    # batch_size = 2000
    # model.train_batch_wise(X_train, Y_train, batch_size)
    
    model.train(X_train, Y_train, X_test, Y_test)

   
    #model.test(X_test, Y_test)
    

def convert_to_vector(y, class_count):
    y_vector = np.zeros([len(y), class_count], dtype=int)
    for row, val in enumerate(y):
        y_vector[row][int(val)] = 1
    return y_vector


main()