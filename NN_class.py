from LR_class import LR, our_cuda_transpose
import numpy as np
from numba import cuda
import array
from time import time

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
        
        start_time = time()

        for iteration in range(self.epochs):

            for lr in self.LRs:
                lr.init_predictions(len(X))
            
            # forward pass
            input = X_device
            for lr in self.LRs:
                lr.forward(input)
                lr.apply_activation_function()
                input = lr.predictions_device # used as input for next iteration

            # backward pass
            prev_loss_device = None
            for i, lr in reversed(list(enumerate(self.LRs))):
                
                loss_device = lr.init_loss(data_size)
                if i == len(self.LRs) - 1:
                    lr.calculate_loss(Y_device, loss_device)
                else:
                    lr_ahead = self.LRs[i + 1]
                    lr._matmul(prev_loss_device, our_cuda_transpose(lr_ahead.theta_transpose_device), loss_device)
                    #get_zero_count_percent(self.LRs[i-1].loss_device)

                input = self.LRs[i-1].predictions_device if i > 0 else X_device
                _ , grad_transpose_device = lr.init_grad_transpose()

                lr.backward(our_cuda_transpose(input), loss_device, grad_transpose_device) #calculate grad_tranpose
                lr.step(self.eta/data_size, grad_transpose_device) # update theta_transpose  

                prev_loss_device = loss_device   

            test_accuracy = self.test(X_test, Y_test) 
            end_time = time()
            total_time = end_time - start_time
            print(f"Epoch {iteration+1}: Accuracy = {test_accuracy:.2f}% Time = {total_time:.2f}s Time per epoch = {total_time/(iteration+1):.2f}s")

    def test(self, X, Y):
        data_size = len(X)
        X = np.c_[X, np.ones(data_size)] 
        X_device = cuda.to_device(X)
        
        #self.theta_transpose_device = cuda.to_device(self.min_theta_transpose)
        input = X_device
        for lr in self.LRs:
            lr.init_predictions(data_size)
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
                
                    
def get_zero_count_percent(data_device):
    data = data_device.copy_to_host()
    size = np.size(data)
    zero_count_percent = (size - np.count_nonzero(data))/size * 100
    print("zero count", zero_count_percent, "%")
    return [zero_count_percent, data]
    
def compress_sparse_matrix(matrix_device): 
    zero_count_percent, data = get_zero_count_percent(matrix_device)
    if zero_count_percent < 50:
        return [None, None]
    sparse_matrix_row_wise =[]
    for row in data:
        sparse_matrix_row_wise.append(np.nonzero(row))
    
    sparse_matrix_col_wise = []
    for row in np.transpose(data):
        sparse_matrix_col_wise.append(np.nonzero(row))
    sparse_matrix_col_wise = np.transpose(sparse_matrix_col_wise)
    
    matrix_device = None #hopefully it will be garbage collected 
    return [sparse_matrix_row_wise, sparse_matrix_col_wise]
    