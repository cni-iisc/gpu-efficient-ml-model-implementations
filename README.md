
This repository contains ML model implementations that utilize GPU *more* efficiently compared to Pytorch. The corresponding torch implementation are inside `torch-implementation` directory. 

## Setup

> P.S. if anybody is willing to contribute a requirements.txt for this repo, please feel free to raise a PR 

- The code currently is tested only on Nvidia GPUs. Install [cuda toolkit](https://developer.nvidia.com/cuda-downloads)
- Install python3 & pip
- Install numba, pandas, sklearn using pip for running our implementation
- Install torch, torchvision for running the pytorch implementation

Our implementation run much faster (nearly half the time) compared to pytorch on the same GPU. Easiest way to measure time is to run `time python <file>.py` to see the time reported at the end. 

Config for running each of the files below. 

## Logistic Regression

Set `TEST_WITH_LR` to `True` in `run.py` & execute `python run.py` . Our LR code slighly worse result with normalized inputs. To disable input normalization, set `NORMALIZE_INPUT` to `False` in the same file. The input & output dimension count is specified in `LR_DIMENSIONS`. The input file is hardcoded inside `main()` function. 

`eta` & `iterations` are defined in `__init__` function of `LR_class`. 

The variable `IS_CPU` can be set to True to benchmark against CPU.

The variable `IS_TEST`, if set to TRUE, will terminate after running 1 iteration and will dump intermediate variables into `dump.txt`.

## Neural Network

Same script `run.py` will execute NN code as well. Set `TEST_WITH_LR` as `False` & the NN code will take over. The dimension of input is defined in `NN_LAYER_DIMENSIONS`. Like LR code, the `eta` & `epocs` are defined in `NN_class`. 

NN code, under the hood, use array of LRs with matching dimensions (LR_1 output dimension = LR_2 input dimension). 

### Backpropagation calculations

While backpropagation matrix multiplication calculations are reasonably straight forward in LR, they are not that well documented for NN. 

Denoting last layer as `L`, the kth layer before the last will be denoted as `L-k`. The layer values will look like 
- input to layer $L-k$ is denoted as $a_{L-k-1}$ . output - $a_{L-k}$ .. for last layer, k is 0 & hence the output is $a_L$ 
- To get output of a layer, $a_{L-k} = a_{L-k-1} \times (W_{L-k})^T$
   - $W_{L-k}$ are weights for layer $L-k$. Dimension of $W_{L-k}$ is $n_{L-k} \times n_{L-k-1}$ 
   - The weights transform $n_{L-k-1}$ dimension input to $n_{L-k}$ dimension output. Hence the transpose during matrix multiplication
   - Dimensions of $a_{L-k}$ is $m \times n_{L-k}$. m represent the input/batch size. 
- loss - $L_{L-k}$ . For the last layer, loss is $a_L - Y$ where $Y$ denotes the label matrix.
   - Dimension of $L_{L-k}$ is $m \times n_{L-k}$
- Grad (also called  $\frac{\partial{L}}{\partial{\theta}}$ ) - $G_{L-k}$ 
   - Dimension of $G_{L-k}$ is same as that of $W_{L-k}$ as weights are updated from the grad. 

**Steps for backward propagation will look like as below:**

1. Calculate gradient from the loss & input to the given layer, starting from Layer L & moving backward

$G_{L-k} = (L_{L-k})^T \times a_{L-k-1}$

2. Update weights of the layer from the gradient 

$W_{L-k-updated} = W_{L-k} - 1/m*(eta*G_{L-k})$

3. Calculate loss for previous layer from current layer loss. Remember to do this step *before* updating weights of the current layer.  

$L_{L-k-1} = L_{L-k} \times W_{L-k}$

4. Go back to step 1 & calculate the gradient $G_{L-k-1}$ . Repeat till the first layer. 

**Things to Note**
- Step 1 & 2 is a known thing. The challenge is to find the previous layer gradient from the current gradient. So step 3 is important & novel. $a_l$ are generated & stored during the forward pass which are then utilized to caluclate the $G_l$ from $L_l$ . 

- $L_L = a_L - Y$ because the loss calculation is negative log likelihood & activation is cross entropy. NLL will lead to $a_L * (1 - a_L)$ in denominator which gets cancelled by the derivative of softmax

- If a different activation function than relu is taken for intermediate layers (like softmax), it will generate extra terms in G calculation. For eg. for softmax, you will get extra $a_{l-k} * (1 - a_{l-k})$ factors

