import numpy as np
import h5py
import math
from process import *

FILE_NAME = 'heart.csv'
TARGET_KEY = 'HeartDisease'

# Activation function for the neural network nodes
def sigmoid(a):
    if a > 100:
        return 1
    elif a < -100:
        return -1
    try:
        sig = 1/(1 + math.exp(-a))
        return sig
    except OverflowError:
        print('Overflow Error')

# Derivative of activation function      
def sigmoid_derivative(a):
    if a > 100 or a < -100:
        return 0
    try:
        sig = math.exp(a)/(math.pow(1 + math.exp(a), 2))
        return sig
    except OverflowError:
        print('Overflow Error')

def train(features:np.array, targets:np.array, nodes:int, step:float, threshold:float, max_epochs:int):
    # N = Samples, M = Dimension of input
    N, M = features.shape

    # Number of hidden nodes and batches
    batches = 10
    
    # Initialize batch size
    batch_size = int(np.floor(N / batches))
    
    # Input and hidden 
    input_nodes = np.zeros(M)
    hidden_nodes = np.zeros(nodes)
    output = 1

    # Initialize weights for nodes (i2h = input-to-hidden, h2o = hidden-to-output)
    i2h_weights = np.random.rand(nodes, M) - 0.5
    best_i2h = i2h_weights

    h2o_weights = np.random.rand(nodes+1) - 0.5
    best_h2o = h2o_weights
    # i2h_weights = np.ones((nodes, M))
    # h2o_weights = np.ones(nodes+1)

    train_err = []
    val_err = []
    epochs = 1
    stop = False
    while not stop:
        temp_i2h = np.array(i2h_weights)
        temp_h2o = np.array(h2o_weights)

        # Initialize storage for gradients
        grad_i2h = np.zeros((nodes, M))
        grad_h2o = np.zeros(nodes+1)

        offset = 0
        for b in range(batches): # Iterate over each batch
            for i in range(batch_size):
                # Features and targets of batch
                x = features[offset+i]
                target = targets[offset+i]
                y = np.zeros(nodes+1)
                
                input_nodes = x
                
                # Start forward feeding
                y[0] = 1
                hidden_nodes = np.zeros(nodes)
                for node in range(nodes):
                    for dim in range(M):
                        hidden_nodes[node] += i2h_weights[node, dim] * input_nodes[dim]
                    y[node+1] = sigmoid(hidden_nodes[node])

                output = 0
                output = np.sum(h2o_weights * y)
                # for node in range(nodes+1):
                #     output += h2o_weights[node] * y[node]
                z = sigmoid(output)
                # End of forward feeding
                
                # Output error
                delta = z - target
                
                # Backpropagation start
                # Second layer gradient
                for node in range(nodes+1):
                    if node == 0:
                        grad_h2o[node] += delta * y[node]
                    else:
                        grad_h2o[node] += delta * y[node] * sigmoid_derivative(output)
                                
                # First layer gradient
                for node in range(nodes):
                    for dim in range(M):
                        grad_i2h[node, dim] += delta * h2o_weights[node+1] * sigmoid_derivative(hidden_nodes[node]) * x[dim]
                # End backpropagation        
                
            # Update layers
            i2h_weights -= step * grad_i2h
            h2o_weights -= step * grad_h2o
            
            offset += batch_size
            
        # Appending errors
        temp = testWeights(i2h_weights, h2o_weights, features, targets)
        
        # Tests stopping condition
        update = (np.linalg.norm(i2h_weights - temp_i2h) + 
                  np.linalg.norm(h2o_weights - temp_h2o))
        # print(f'Norm Update = {update}')
        print(f'Current training error = {temp}, Norm Update = {update}')
        stop = update < threshold or epochs >= max_epochs
        epochs += 1
    return i2h_weights, h2o_weights
                
            
def testWeights(weights1, weights2, features, targets):
    N, M = features.shape
    nodes = weights1.shape[0]
    error = 0
    for n in range(N):
        x = features[n]
        actual = targets[n]
        
        y = np.zeros(nodes+1)
        
        y[0] = 1
        for node in range(nodes):
            a = 0
            for dim in range(M):
                a += weights1[node, dim] * x[dim]
            y[node+1] = sigmoid(a)
        
        a = 0
        a = np.sum(weights2 * y)
        z = sigmoid(a)
        
        error += math.pow(z - actual, 2)
    
    return error

if __name__ == "__main__":
    num_nodes = input('Number of hidden nodes: ')
    data = process_data(FILE_NAME)
    targets, features = extract_data(data, TARGET_KEY)
    train(features[:600], targets, int(num_nodes), 0.0001, 0.0001, 200)