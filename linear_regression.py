import numpy as np
import h5py
import math
from process import *

FILE_NAME = 'heart.csv'
TARGET_KEY = 'HeartDisease'

def sigmoid(theta, phi):
    a = np.matmul(theta, phi) # Matmul automatically transposes theta
    if a > 100:
        return 1
    elif a < -100:
        return 0
    try:
        sig = 1/(1 + math.exp(-a))
        return sig
    except OverflowError:
        print(a)

def gradient(features, thetas, target):
    err = 0
    N, M = features.shape
    for i in range(N):
        phi = features[i]

        prediction = sigmoid(thetas, phi)
        t = target[i]
        err += (prediction-t)*phi
    return thetas + err

def descent(features, eta, threshold, thetas, target):
    grad = gradient(features, thetas, target)
    while np.linalg.norm(grad, ord=2) > threshold:
        err = np.linalg.norm(grad, ord=2)

        temp_thetas = thetas - eta*grad
        errAfter = np.linalg.norm(gradient(features, temp_thetas, target), ord=2)

        print(f'Current Error: {err}')

        # Adjust eta
        if errAfter < err:
            eta *= 1.7
        elif errAfter >= err:
            eta /= 2
        thetas -= eta*grad
        grad = gradient(features, thetas, target)

    return thetas
        
def calculate_error(features, thetas, target):
    N, M = features.shape
    err = np.zeros(N)
    for i in range(N):
        # Calculate the probability of sample being 1
        phi = features[i]
        p = sigmoid(thetas, phi)
        actual = target[i]

        # Compare probability with actual label
        if p > 0.5 and actual != 1:
            err[i] = 1
        elif p <= 0.5 and actual != 0:
            err[i] = 1
    return np.sum(err) / N

def objective(thetas, target, features, S):
    err = 0
    offset = 0.0000000000001
    N, M = features.shape
    for i in range(N):
        phi = features[i]
        actual = target[i]
        prediction = sigmoid(thetas, phi)

        err += actual*math.log(abs(prediction+offset)) + (1-actual)*math.log(abs(1-prediction+offset))
    minErr = (M/2)*math.log(2*math.pi) + .5*np.matmul(np.transpose(thetas), np.matmul(S, thetas))
    return minErr

def train(train_features, train_targets):
    N, M = train_features.shape
    # Initialize variables
    thetas = np.ones(M)
    eta = 0.00001
    S = np.identity(M)
    threshold = 100

    thetas = descent(train_features, eta, threshold, thetas, train_targets)
    min_err = objective(thetas, train_targets, train_features, S)
    print(f'Minimum error from objective function = {min_err}')
    return thetas

if __name__ == "__main__":
    data = process_data(FILE_NAME)
    N, M = data.shape
    targets, features = extract_data(data, TARGET_KEY)
    features = features[:,:10]
    #features = np.delete(features, [3], axis=1)#[:,3:]
    thetas = train(features[:600], targets[:600])
    trainError = calculate_error(features[:600], thetas, targets[:600])
    print(f'Training set error = {trainError}')

    # Test set
    testError = calculate_error(features[600:], thetas, targets[600:])
    print(f'Testing set error = {testError}')