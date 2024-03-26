from LR_class import LR, our_cuda_transpose
import numpy as np
from numba import cuda
import array

class NN():
    min_total_loss = float('inf')
    
    def __init__(self, layers_feature_count=(28*28, 10000, 10), eta=0.01, epochs=10):
        self.epochs = epochs
        self.eta = eta 
        
        print("layers_feature_count ", layers_feature_count)
        self.LRs = []
        for i, feature_count in enumerate(layers_feature_count):
            if i == 0: continue
            if i != len(layers_feature_count) - 1:
                self.LRs.append(LR(layers_feature_count[i-1], feature_count, "relu", False))
            else:
                self.LRs.append(LR(layers_feature_count[i-1], feature_count, "softmax", True)) #choose softmax for last layer
        
        print(f"Created {len(self.LRs)} LRs")

    def train(self, X, Y, X_test, Y_test):
            
        data_size = len(X)
        X = np.c_[X, np.ones(data_size)] #appending ones for the bias

        X_device = cuda.to_device(X)
        Y_device = cuda.to_device(Y)
        
        
        boosted_iteration = 0
        training_accuracy_array = []

        for iteration in range(self.epochs):

            for lr in self.LRs:
                lr.init_after_data(len(X))
            
            # forward pass
            input = X_device
            for lr in self.LRs:
                lr.forward(input)
                lr.apply_activation_function()
                input = lr.predictions_device # used as input for next iteration

            # backward pass
            for i, lr in reversed(list(enumerate(self.LRs))):

                if i == len(self.LRs) - 1:
                    lr.calculate_loss(Y_device)
                    # loss_device_host = lr.loss_device.copy_to_host()
                    # loss = np.sum(np.abs(loss_device_host))
                    # training_accuracy = ((data_size - loss/2) / data_size) * 100
                    # print("training accuracy ", training_accuracy , "% at iteration ", iteration)
                    
                    # if len(training_accuracy_array) > 5:
                    #     delta = training_accuracy - training_accuracy_array[iteration - 5]
                    #     if training_accuracy > 80 and delta > 0 and delta < 1 and iteration - boosted_iteration > 5:
                    #         print("boosting\n")
                    #         self.eta *= 1.1
                    #         boosted_iteration = iteration
                    
                    # training_accuracy_array.append(training_accuracy)

                # loss for previous LR .. calculate this before updating theta of current LR 
                if i > 0:
                    lr._matmul(lr.loss_device, our_cuda_transpose(lr.theta_transpose_device), self.LRs[i-1].loss_device)

                input = self.LRs[i-1].predictions_device if i > 0 else X_device
                lr.backward(our_cuda_transpose(input)) #calculate grad_tranpose
                lr.step(self.eta/data_size) # update theta_transpose    

            test_accuracy = self.test(X_test, Y_test) 
            print("test accuracy ", test_accuracy, "in iteration ", iteration)

    def test(self, X, Y):
        data_size = len(X)
        X = np.c_[X, np.ones(data_size)] 
        X_device = cuda.to_device(X)
        
        #self.theta_transpose_device = cuda.to_device(self.min_theta_transpose)
        input = X_device
        for lr in self.LRs:
            lr.init_after_data(data_size)
            lr.forward(input)
            lr.apply_activation_function()
            input = lr.predictions_device

        Y_hat = input.copy_to_host()
    
        correct_prediction_count = 0
      
        for index, row in enumerate(np.argmax(Y_hat, 1)):
            if Y[index, row] == 1:
                #print(Y[index], Y_hat[index])
                #print(index)
                correct_prediction_count += 1
        #print(correct_prediction_count)
        accuracy = (correct_prediction_count * 100)/len(Y)
        return accuracy 
        #print("Accuracy ", accuracy)   
                
                    
