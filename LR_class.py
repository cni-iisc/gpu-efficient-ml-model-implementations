import math
from numba import cuda
from numba.cuda.kernels.transpose import transpose as cuda_transpose
import time
import numpy as np
import os


TPB = 16 #calculated from the input size .. the value need to be predefined for CUDA multiplication 
BLOCK_SIZE = 256 # for some reason, the code fails for > 256 values .. dont know why
IS_TEST = False
IS_CPU = False
IS_NUMBA_ENABLE_CUDASIM = os.getenv('NUMBA_ENABLE_CUDASIM', False)
if IS_NUMBA_ENABLE_CUDASIM:
    IS_TEST = True
f = None

class LR():
    min_total_loss = float('inf')
    
    
    def __init__(self, input_dimension_count, output_dimension_count, activation_function = "softmax", is_last_layer = True, theta_choice = "positive_negative", eta=6, iterations=1000):
        self.input_dimension_count = input_dimension_count
        self.output_dimension_count = output_dimension_count
        self.eta = eta
        self.iterations = iterations
        self.is_last_layer = is_last_layer

        #self.theta = (np.random.rand(self.class_count, self.data_dimension_count + 1) - 0.5) * 2
        if theta_choice == "only_positive":
            self.theta = np.random.rand(self.output_dimension_count, self.input_dimension_count + 1) 
        else:
            self.theta = (np.random.rand(self.output_dimension_count, self.input_dimension_count + 1) - 0.5) * 2

        #self.theta = np.zeros([self.class_count, self.data_dimension_count + 1]) 

        #self.theta_device = cuda.to_device(self.theta)
        self.theta_transpose = np.transpose(self.theta)
        self.theta_transpose_device = cuda.to_device(self.theta_transpose)
        self.min_theta_transpose = None

        self.activation_function = activation_function

        self.grad_transpose = np.zeros([self.input_dimension_count + 1, self.output_dimension_count])
        self.grad_transpose_device =  cuda.to_device(self.grad_transpose)
        
        print ("LR initialized with ", self.input_dimension_count,"x",self.output_dimension_count, "activation function ", activation_function, "theta_choice ", theta_choice)
        
    def init_after_data(self, data_size):
        self.predictions = np.zeros([data_size, self.output_dimension_count])
        if not self.is_last_layer:
            self.predictions = np.c_[self.predictions, np.ones(data_size)]
            
        self.predictions_device = cuda.to_device(self.predictions)
        
        self.loss = np.zeros([data_size, self.output_dimension_count])
        self.loss_device = cuda.to_device(self.loss)

    def train_batch_wise(self, X, Y, batch_size):
        data_size = len(X)
        # X = np.c_[X, np.ones(data_size)]
        for i in range(0, data_size, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            self.train(X_batch, Y_batch)
            # break

    def train(self, X, Y, X_test, Y_test):
        if IS_TEST:
            f = open("dump.txt", "w")
            
        data_size = len(X)
        X = np.c_[X, np.ones(data_size)] #appending ones for the bias

        start_time = time.time()
        X_device = cuda.to_device(X)
        X_transpose_device = cuda.to_device(np.transpose(X))
        Y_device = cuda.to_device(Y)
        
        self.init_after_data(data_size)
        
        #print("Elapsed time transferring data to cuda: ", time.time() - start_time)
    
        for i in range(self.iterations):
            self.forward(X_device)
            self.predictions = self.forward_cpu(X, self.predictions, self.predictions_device)

            self.apply_activation_function()
            self.predictions = self.apply_activation_function_cpu(self.predictions, self.predictions_device)
            #predictions_device = cuda.to_device(predictions)

            self.calculate_loss(Y_device)
            self.loss = self.calculate_loss_cpu(self.predictions, Y, self.loss_device)

            loss_device_host = self.loss_device.copy_to_host()
            total_loss = np.sum(np.abs(loss_device_host))
            #print("total loss device ", np.sum(np.abs(loss_device_host)))
            if IS_CPU == True: print("total loss cpu ", np.sum(np.abs(self.loss)))
            
            if self.min_total_loss > total_loss:
                self.min_total_loss = total_loss
                self.min_theta_transpose = self.theta_transpose_device.copy_to_host()
                training_accuracy = 100 - ((self.min_total_loss/2) / data_size * 100)
                print("max training accuracy ", training_accuracy, " at iteration ", i)

            self.backward(X_transpose_device)
            self.grad_transpose = self.backward_cpu(self.loss, np.transpose(X), self.grad_transpose, self.grad_transpose_device )
        
            self.step(self.eta/data_size)
            self.step_cpu(self.grad_transpose)

            # if IS_TEST and i == 1 :
            #     print(f"breaking at iteration {i} due to IS_TEST set")
            #     break
        #print("iterations done ", self.iterations)
        
        if IS_TEST:
            f.close()
            
        self.test(X_test, Y_test)


    def test(self, X, Y):
        data_size = len(X)
        X = np.c_[X, np.ones(data_size)] 
        X_device = cuda.to_device(X)
        
        self.theta_transpose_device = cuda.to_device(self.min_theta_transpose)
        self.init_after_data(data_size)
        self.forward(X_device)
        self.apply_activation_function()
        Y_hat = self.predictions_device.copy_to_host()
    
        correct_prediction_count = 0
      
        for index, row in enumerate(np.argmax(Y_hat, 1)):
            if Y[index, row] == 1:
                #print(Y[index], Y_hat[index])
                #print(index)
                correct_prediction_count += 1
        print(correct_prediction_count)
        accuracy = (correct_prediction_count * 100)/len(Y)
        print("Testing accuracy ", accuracy)


    def forward(self, X_device):
        self._matmul(X_device, self.theta_transpose_device, self.predictions_device)

    def forward_cpu(self, X, predictions, predictions_device):
        if IS_CPU == False: return 
    
        predictions =  np.matmul(X , self.theta_transpose)
        dump_device_variable("predictions ", predictions, predictions_device)
        return predictions
    
    def apply_activation_function(self):
        data = self.predictions_device
        if self.activation_function == "softmax":
            threadsperblock_x =  min(BLOCK_SIZE, len(data[0]))
            threadsperblock = (threadsperblock_x, 1)
            blockspergrid_x = math.ceil(data.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(data.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            #print("softmax ", threadsperblock, blockspergrid)
            calculate_softmax[blockspergrid, threadsperblock](data)
        elif self.activation_function == "relu":
            threadsperblock = get_thread_block_size(data)
            blockspergrid_x = math.ceil(data.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(data.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            #print("relu ", threadsperblock, blockspergrid)
            calculate_relu[blockspergrid, threadsperblock](data)
    
    def apply_activation_function_cpu(self, data, data_device):
        if IS_CPU == False: return 

        if self.activation_function == "softmax":
            def sigmoid(row):
                #print(row)
                max_index = np.argmax(row)
                for i, val in enumerate(row):
                    row[i] = 1 if i == max_index else 0
                return row
            np.apply_along_axis(sigmoid, axis=1, arr=data )
        elif self.activation_function == "relu":
            data = data * (data > 0)
        
        dump_device_variable("activation ", data, data_device)
        return data


    def calculate_loss(self, Y_device):
        self._copy_array(self.loss_device, self.predictions_device)
        self._matsubtract(self.loss_device, Y_device)  
    
    def calculate_loss_cpu(self, predictions, Y, loss_device):
        if IS_CPU == False: return 

        loss = np.subtract(predictions, Y)
        dump_device_variable("loss ", loss, loss_device)
        return loss

    def backward(self,  X_transpose_device):
        self._matmul(X_transpose_device, self.loss_device, self.grad_transpose_device)

    def backward_cpu(self, loss, X_transpose, grad_transpose, grad_transpose_device):
        if IS_CPU == False: return 

        grad_transpose = np.matmul(X_transpose, loss)
        dump_device_variable("grad ", grad_transpose, grad_transpose_device )
        return grad_transpose

    def step(self, eta):
        #doe_L_by_doe_theta_device = our_cuda_transpose(doe_L_by_doe_theta_transpose_device)
        self._matmulWithConstant(self.grad_transpose_device, eta)
        self._matsubtract(self.theta_transpose_device, self.grad_transpose_device)

        #self.theta_transpose_device = None
        #self.theta_transpose_device = our_cuda_transpose(self.theta_device)

    def step_cpu(self, grad_transpose):
        if IS_CPU == False: return 

        self.theta_transpose -= grad_transpose * self.eta
        dump_device_variable("theta transpose ", self.theta_transpose, self.theta_transpose_device)
        

    def _matmul(self, A, B, C):
        threadsperblock = (TPB, TPB) # threads per block need to be square
        blockspergrid_x = math.ceil(C.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(C.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        #print("_matmul", threadsperblock, blockspergrid)
        matmul_gpu_slow[blockspergrid, threadsperblock](A, B, C)


    def _copy_array(self, dest, source):
        threadsperblock = get_thread_block_size(source)
        blockspergrid_x = math.ceil(source.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(source.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        #print("copy_array ", threadsperblock, blockspergrid)
        copy_array[blockspergrid, threadsperblock](dest, source)

    def _matsubtract(self, A, B):
        threadsperblock = get_thread_block_size(A)
        blockspergrid_x = math.ceil(A.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(A.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        #print("_matsubtract ", threadsperblock, blockspergrid)
        matsubtact_gpu[blockspergrid, threadsperblock](A, B)
    
    def _matmulWithConstant(self, A, const):
        threadsperblock = get_thread_block_size(A)
        blockspergrid_x = math.ceil(A.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(A.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        #print("_matmulWithConstant  ", threadsperblock, blockspergrid)
        matmulWithConstant[blockspergrid, threadsperblock](A, const)

    def _hadamardmul(self,A,B,C):
        threadsperblock = get_thread_block_size(A)
        blockspergrid_x = math.ceil(A.shape[0]/threadsperblock[0])
        blockspergrid_y = math.ceil(A.shape[1]/threadsperblock[1])
        blockspergrid = (blockspergrid_x,blockspergrid_y)
        hadamardMult_gpu[blockspergrid, threadsperblock](A,B,C)
        
def get_thread_block_size(matrix):
    smaller_side = min(matrix.shape[0], matrix.shape[1])
    if smaller_side >= BLOCK_SIZE:
        return (TPB, TPB)
    
    
    larger_side =  math.floor(BLOCK_SIZE / smaller_side)
    if smaller_side == matrix.shape[0]:
        return(smaller_side, min(larger_side, matrix.shape[1]))
    else:
        return ( min(larger_side, matrix.shape[0]), smaller_side)
    
def our_cuda_transpose(matrix):
    if IS_NUMBA_ENABLE_CUDASIM:
        return np.transpose(matrix)
    else:
        return cuda_transpose(matrix)

def dump_device_variable(text, variable, variable_device):
    if IS_TEST == False:
        return 
    
    if IS_NUMBA_ENABLE_CUDASIM:
        print(text, variable)
        return
    
    variable_host = variable_device.copy_to_host()
    abs_difference = np.sum(np.abs(np.subtract(variable, variable_host)))
    if abs_difference < 10: return
    
    print(text,": ",abs_difference, "\n", variable[0:10,0:10],"\n",variable_host[0:10,0:10] )

    f.write(text + ":" + str(abs_difference) + "\n")
    np.savetxt(f, variable, fmt='%1.2f')
    f.write("\n---\n")
    np.savetxt(f, variable_host, fmt='%1.2f')
    f.write("\n-------------\n")

    if abs_difference > 10:
        raise Exception("mayday mayday mayday")

@cuda.jit
def calculate_relu(data):
    x, y = cuda.grid(2)
    if  x < data.shape[0] and y < data.shape[1] and data[x,y] < 0:
        data[x,y] = 0

@cuda.jit
def copy_array(dest,source):
    x, y = cuda.grid(2)
    if x < source.shape[0] and y < source.shape[1]:
        dest[x,y] = source[x,y]

@cuda.jit
def matmulWithConstant(A, const):
    x, y = cuda.grid(2)
    if x < A.shape[0] and y < A.shape[1]:
        A[x,y] *= const

@cuda.jit(debug=True, opt=False)
def matsubtact_gpu(A, B):
    x, y = cuda.grid(2)
    if x < A.shape[0] and y < A.shape[1]:
        A[x,y] = A[x,y] - B[x,y]

@cuda.jit
def hadamardMult_gpu(A,B,C):
    x, y = cuda.grid(2)
    if A.shape == B.shape:
        if x < A.shape[0] and y < A.shape[1]:
            C[x,y] = A[x,y]*B[x,y]

@cuda.jit
def calculate_softmax(data):
    x, y = cuda.grid(2)
    max = np.float64(0.)
    pos = 0
    if x < data.shape[0]:
        for i in range(data.shape[1]):
            if i == 0:
                max = data[x,0]
            elif data[x,i] > max:
                max = data[x,i]
                pos = i

        for i in range(data.shape[1]):
            if i == pos:
                data[x, i] = 1
            else:
                data[x, i] = 0



@cuda.jit
def matmul_gpu_slow(A, B, C):
    x, y = cuda.grid(2)
    if x < A.shape[0] and y < B.shape[1]:
        sum = np.float64(0.)
        for i in range(A.shape[1]):
            sum += A[x,i] * B[i,y]
        C[x,y] = np.float64(sum)
   
        
@cuda.jit
def matmul_gpu(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=np.float64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=np.float64)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = np.float64(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = 0
        sB[tx, ty] = 0
        
        if x < A.shape[0] and (ty+i*TPB) < A.shape[1]:
            sA[tx, ty] = A[x, ty + i * TPB]
        
        if y < B.shape[1] and (tx+i*TPB) < B.shape[0]:
            sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if x < C.shape[0] and y < C.shape[1]:
        C[x, y] = np.float64(tmp)

