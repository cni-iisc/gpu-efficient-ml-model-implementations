from numba import cuda
from numba.cuda.kernels.transpose import transpose as cuda_transpose
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from random import randint
import math
import os
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler

TPB = 16
MAX_THREAD_COUNT_PER_SM = 256
IS_TEST = False
IN_CPU = False
IS_NUMBA_ENABLE_CUDASIM = os.getenv('NUMBA_ENABLE_CUDASIM', False)

if IS_TEST:
    f = open("dump.txt", "w")

def main():
    mnist = fetch_openml('mnist_784', parser= 'auto')
    data, target = mnist.data, mnist.target
    print("Dataset Loaded...")
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42,test_size=0.142857)
    num_classes = 10
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Normalizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #print(data, target)

    # Loading new dataset
    # X_train, y_train, X_test, y_test = datasetLoading()
  
    Y_train = convert_to_vector(y_train, num_classes)
    Y_test = convert_to_vector(y_test, num_classes)
    print("Dataset Configured...")
    model = NeuralNetwork(X_train.shape[0], X_test.shape[0], X_train.shape[1], num_classes, hidden_layers=[1000,256], activation='relu', loss='crossEntropy', iterations=100, eta=0.01)
    print("model Loaded...")
    # For running dataset using batches uncomment the lines below
    # batch_size = 2000
    # model.train_batch_wise(X_train, Y_train, batch_size)
    
    model.train(X_train, Y_train)
    model.test(X_test, Y_test)

def test():
    data_size = 3
    data_dimension_size = 3
    num_classes = 3
    X_train = np.ones([data_size, data_dimension_size])
    Y_train = np.zeros([data_size,num_classes])
    for row, row_y in enumerate(Y_train):
        idx = randint(0, num_classes - 1)
        Y_train[row][idx] = 1
    model = NeuralNetwork(data_size, num_classes)
    model.train(X_train, Y_train)

class NeuralNetwork():
    def __init__(self, n_samples_train, n_samples_test, n_features, n_classes, hidden_layers=(), activation='relu', loss='crossEntropy', iterations=100, eta=0.5):
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        self.n_features = n_features
        self.n_classes = n_classes
        self.input_layer_shape = (n_samples_train, n_features)
        self.output_layer_shape = (n_samples_test, n_classes)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.loss = loss
        self.iterations = iterations
        self.weights, self.weights_device, self.grads_device_transpose = self.init_weights()
        # self.A, self.A_device = self.init_outputs()
        self.learning_rate = eta

    def init_weights(self):
        thetas = []
        thetas_device = []
        grad_devices = []
        for i in range(len(self.hidden_layers)+1):
            if i == 0:
                t = (np.random.rand(self.input_layer_shape[1]+1, self.hidden_layers[i])-0.5)*2
                thetas.append(t)
                thetas_device.append(cuda.to_device(t))
                grad_devices.append(cuda.to_device(np.zeros(t.T.shape)))
            elif i == len(self.hidden_layers):
                t = (np.random.rand(self.hidden_layers[i-1], self.output_layer_shape[1])-0.5)*2
                thetas.append(t)
                thetas_device.append(cuda.to_device(t))
                grad_devices.append(cuda.to_device(np.zeros(t.T.shape)))
            else:
                t = (np.random.rand(self.hidden_layers[i-1]+1, self.hidden_layers[i])-0.5)*2
                thetas.append(t)
                thetas_device.append(cuda.to_device(t))
                grad_devices.append(cuda.to_device(np.zeros(t.T.shape)))
        return thetas,thetas_device, grad_devices
    
    def init_outputs(self, input_size):
        A = []
        A_device = []
        for i in range(len(self.hidden_layers)+1):
            if i == len(self.hidden_layers):
                a = np.zeros((input_size, self.output_layer_shape[1]))
                A.append(a)
                A_device.append(cuda.to_device(a))
            else:
                a = np.zeros((input_size, self.hidden_layers[i]))
                a = np.c_[a, np.ones(input_size)]
                A.append(a)
                A_device.append(cuda.to_device(a))
        return A, A_device

    def train(self,X,y):
        X = np.c_[X, np.ones(self.input_layer_shape[0])]
        Y_device = cuda.to_device(y)
        flag = True
        # A_device_prev = None
        for i in range(self.iterations):
            A, A_device = self.forward(X,self.n_samples_train)
            # A = self.forward_cpu(X,self.n_samples_train)
            # dump_device_variable("Outputs", A, A_device)
            loss_device = self.calculate_loss(A_device[len(A_device)-1], Y_device)
            # loss = self.calculate_loss_cpu(A[len(A)-1], y)
            loss = loss_device.copy_to_host()
            loss = np.sum(np.abs(loss))
            if loss < 0.30 and flag:
                self.learning_rate *= 3
                flag = False
            print(f"Loss at iteration {i} is {loss}")
            # print(f"Loss at iteration {i} is {np.sum(np.abs(loss))}")
            self.backward(cuda.to_device(X),loss_device,A_device)
            # self.backward_cpu(X,loss,A)

            # debug(A_device_prev, A_device)
            # A_device_prev = A_device
            # if IS_TEST and i == 1:
            #     break
        
        if IS_TEST:
            f.close()
        return
        
    def test(self,X,y):
        X = np.c_[X, np.ones(X.shape[0])]
        A, A_device = self.forward(X,self.n_samples_test)
        # print(len(output))
        Y_hat = A_device[len(A_device)-1].copy_to_host()

        correct_prediction_count = 0

        for index, row in enumerate(np.argmax(Y_hat, 1)):
            if y[index, row] == 1:
                #print(Y[index], Y_hat[index])
                #print(index)
                correct_prediction_count += 1
        print(correct_prediction_count)
        accuracy = (correct_prediction_count * 100)/len(y)
        print(accuracy)

    def forward(self, X, data_size):
        A, A_device = self.init_outputs(data_size)
        if IN_CPU:
            A = self.forward_cpu(X, data_size)
        for i, weight_device in enumerate(self.weights_device):
            if i == 0:
                self._matmul(cuda.to_device(X), weight_device, A_device[i])
                self.apply_relu(A_device[i])
            elif i == len(self.weights_device)-1:
                self._matmul(A_device[i-1], weight_device, A_device[i])
                self.apply_softmax(A_device[i])
            else:
                self._matmul(A_device[i-1], weight_device, A_device[i])
                self.apply_relu(A_device[i])
            # print(i, A_device[i])
        return A, A_device
    
    def forward_cpu(self, X, data_size):

        def sigmoid(row):
            #print(row)
            max_index = np.argmax(row)
            for i, val in enumerate(row):
                row[i] = 1 if i == max_index else 0
            return row

        A, A_device = self.init_outputs(data_size)
        for i, weight in enumerate(self.weights):
            if i == 0:
                A[i] = np.matmul(X, weight)
                A[i] = np.maximum(A[i], 0)
            elif i == len(self.weights)-1:
                A[i] = np.matmul(A[i-1], weight)
                A[i] = np.apply_along_axis(sigmoid, axis=1, arr=A[i])
            else:
                A[i] = np.matmul(A[i-1], weight)
                A[i] = np.maximum(A[i], 0)
        return A

    # Validate this function with equations once again.
    def backward(self,X_device,loss_device,A_device):
        loss_device_next = None
        loss_device_next_prev = None
        for i in range(len(self.weights_device)-1, -1, -1):
            # Calculate gradients
            if i == len(self.weights_device)-1:
                loss_device_next = cuda.to_device(np.zeros(A_device[i-1].shape))
                self._matmul(cuda_transpose(loss_device), A_device[i-1], self.grads_device_transpose[i])
                self._matmul(loss_device, cuda_transpose(self.weights_device[i]), loss_device_next)
            elif i == 0:
                self._matmul(cuda_transpose(loss_device_next), X_device, self.grads_device_transpose[i])
            else:
                self._matmul(cuda_transpose(loss_device_next), A_device[i-1], self.grads_device_transpose[i])
                loss_device_next = cuda.to_device(np.zeros(A_device[i-1].shape))
                self._matmul(loss_device_next_prev, cuda_transpose(self.weights_device[i]), loss_device_next)

            # Update weights
            self._matmulWithConstant(self.grads_device_transpose[i], self.learning_rate)
            self._matsubtract(self.weights_device[i], cuda_transpose(self.grads_device_transpose[i]))
            loss_device_next_prev = loss_device_next
        return
    
    def backward_cpu(self,X,loss,A):
        if not IN_CPU: return
        loss_next = None
        loss_next_prev = None
        for i in range(len(self.weights)-1, -1, -1):
            # Calculate gradients
            if i == len(self.weights)-1:
                grads = np.matmul(A[i-1].T, loss)
                loss_next = np.matmul(loss, self.weights[i].T)
            elif i == 0:
                grads = np.matmul(X.T, loss_next)
            else:
                grads = np.matmul(A[i-1].T, loss_next)
                loss_next = np.matmul(loss_next_prev, self.weights[i].T)

            # Update weights
            self.weights[i] = np.subtract(self.weights[i], grads * self.learning_rate)
            loss_next_prev = loss_next
        
        dump_device_variable("weights", self.weights, self.weights_device)

        return

    def calculate_loss(self, output_layer, Y_device):
        self._matsubtract(output_layer, Y_device)
        self._matmulWithConstant(output_layer, 1/self.n_samples_train)
        return output_layer
    
    def calculate_loss_cpu(self, output_layer, Y):
        return np.subtract(output_layer, Y)
    
    def apply_relu(self, data):
        if self.activation == 'relu':
            threadsperblock_y =  min(MAX_THREAD_COUNT_PER_SM, len(data[0]))
            threadsperblock = (1, threadsperblock_y)
            blockspergrid_x = math.ceil(data.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(data.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            calculate_relu[blockspergrid, threadsperblock](data)
        
    def apply_softmax(self, data):
        threadsperblock_y =  min(MAX_THREAD_COUNT_PER_SM, len(data[0]))
        threadsperblock = (1, threadsperblock_y)
        blockspergrid_x = math.ceil(data.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(data.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        calculate_softmax[blockspergrid, threadsperblock](data)

    def _matmul(self, A, B, C):
        threadsperblock = (TPB, TPB) # threads per block need to be square
        blockspergrid_x = math.ceil(C.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(C.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        matmul_gpu_slow[blockspergrid, threadsperblock](A, B, C)

    def _matsubtract(self, A, B):
        #print(A.shape, B.shape)
        threadsperblock = get_thread_block_size(A)
        blockspergrid_x = math.ceil(A.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(A.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        #print(A,B)
        matsubtact_gpu[blockspergrid, threadsperblock](A, B)

    def _matmulWithConstant(self, A, const):
        threadsperblock = get_thread_block_size(A)
        blockspergrid_x = math.ceil(A.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(A.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        matmulWithConstant[blockspergrid, threadsperblock](A, const)

def dump_device_variable(text, variable, variable_device):
    if IS_TEST == False:
        return 
    
    
    for i, variable_ in enumerate(variable):
        if IS_NUMBA_ENABLE_CUDASIM:
            print(text, variable_)
            return
        variable_host = variable_device[i].copy_to_host()
        abs_difference = np.sum(np.abs(np.subtract(variable_, variable_host)))
        # if abs_difference < 10: return
        
        print(text,": ", "For iteration: ", i,abs_difference, "\n", variable_[0:10,0:10],"\n",variable_host[0:10,0:10] )

        f.write(text + ":" + str(abs_difference) + "\n")
        np.savetxt(f, variable_[0:100,0:100], fmt='%1.2f')
        f.write("\n---\n")
        np.savetxt(f, variable_host[0:100,0:100], fmt='%1.2f')
        f.write("\n-------------\n")

        if abs_difference > 20:
            raise Exception("mayday mayday mayday")

def get_thread_block_size(matrix):
    smaller_side = min(matrix.shape[0], matrix.shape[1])
    if smaller_side >= MAX_THREAD_COUNT_PER_SM:
        return (16,16)
    
    larger_side =  math.floor(MAX_THREAD_COUNT_PER_SM / smaller_side)
    if smaller_side == matrix.shape[0]:
        return(smaller_side, min(larger_side, matrix.shape[1]))
    else:
        return ( min(larger_side, matrix.shape[0]), smaller_side)

def debug(prev_arr, arr):
    if prev_arr is None:
        return
    for i , val in enumerate(arr):
        val = val.copy_to_host()
        prev_arr[i] = prev_arr[i].copy_to_host()
        if np.sum(np.abs(val - prev_arr[i])) == 0:
            print("No change in matrix ", i)
            return

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


# @cuda.jit
# def calculate_relu(data):
#     x, y = cuda.grid(2)
#     if x < data.shape[0] and y < data.shape[1]:
#         data[x,y] = max(0, data[x,y])

@cuda.jit
def calculate_relu(data):
    x, y = cuda.grid(2)
    if  x < data.shape[0] and y < data.shape[1] and data[x,y] < 0:
        data[x,y] = 0

@cuda.jit
def calculate_softmax(data):
    x, y = cuda.grid(2)

    vector = cuda.shared.array(shape=(10), dtype=np.float64) # using float64 instead of float64 does not lead to sum of final values to 1. 
    sum = cuda.shared.array(shape=(1), dtype=np.float64)
    sum[0] = 0.
    max = cuda.shared.array(shape=(1), dtype=np.float64)

    if x < data.shape[0] and y < data.shape[1]:
        vector[y] = data[x,y]

        cuda.syncthreads()
        #cuda.atomic.max(max, 0, vector[y])
        if y == 0:
            max[0] = vector[0]
            for i, val in enumerate(vector):
                max[0] = val if val > max[0] else max[0]
                sum[0] += val
        cuda.syncthreads()
        
        if sum[0] == 0:
            if y == 0:
                vector[y] = 1
            else:
                vector[y] = 0
        else:
            if vector[y] == max[0]:
                vector[y] = 1
            else:
                vector[y] = 0
    # if x == 0 and y == 0:
    #     print("max ", max[0])
    
    #if max[0] > 0:
    # vector[y] = vector[y] - max[0]
    # vector[y] = math.exp(vector[y])
    # # if x == 0 :
    # #     print("vector[y] - max[0] ", val, " vector[y] after exp ", vector[y])
 
    # cuda.atomic.add(sum, 0, vector[y])

    # cuda.syncthreads()
    # if x == 0 and y == 0:
    #     print("sum ", sum[0])

    # if sum[0] != 0.: # if sum is 0, all values will be 0 & we dont need to update them 
    #     vector[y] = vector[y]/sum[0]
    #if x < data.shape[0] and y < data.shape[1]:
        data[x,y] = vector[y]


@cuda.jit
def matmul_gpu_slow(A, B, C):
    x, y = cuda.grid(2)
    if x < C.shape[0] and y < C.shape[1]:
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

def convert_to_vector(y, class_count):
    y_vector = np.zeros([len(y), class_count], dtype=int)
    for row, val in enumerate(y):
        y_vector[row][int(val)] = 1
    return y_vector


if __name__ == "__main__":
    main()
    #test()
