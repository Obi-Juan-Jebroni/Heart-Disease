import numpy as np
import sys
import matplotlib.pyplot as plt
from process import *

FILE_NAME = 'heart.csv'
CATEGORICAL_COLUMNS = ['FastingBS',
                       'Sex_F',
                       'ChestPainType_ASY',
                       'ChestPainType_ATA',
                       'ChestPainType_NAP',
                       'ChestPainType_TA',
                       'RestingECG_LVH',
                       'RestingECG_Normal',
                       'RestingECG_ST',
                       'ExerciseAngina_N',
                       'ExerciseAngina_Y',
                       'ST_Slope_Down',
                       'ST_Slope_Flat',
                       'ST_Slope_Up']
TARGET_KEY = 'HeartDisease'

def naivebayes(targets, features):
    # Keep track of priors and pixel probabilities for each class
    classes = 2
    count = np.zeros(classes, dtype=int)
    prob = np.zeros(classes, dtype=float)

    # Iterate through all images and pixels
    N = features.shape
    for i in range(targets.size):
        img_class = targets[i]
        prob[img_class] += features[i]
        count[img_class] += 1

    # Calculate class priors
    priors = count / N

    # Calculate pixel probabilities for each class
    for i in range(classes):
        prob[i] = prob[i] / count[i]

    return priors, prob


# Predict classification results based on trained probabilities
def predict(prob, data):
    N, categories = data.shape
    classes, M = prob.shape
    predictions = np.zeros(N, dtype=int)
    for s in range(N):
        x = data[s]
        max = -sys.maxsize
        best = None
        for cl in range(classes):
            offset = .00000000000000000001
            cmp = np.sum(x*np.log(prob[cl]+offset) + (1-x)*np.log(1-prob[cl]+offset))
            if cmp > max:
                max = cmp
                best = cl
        predictions[s] = best
    return predictions

# Calculate error between predictions and actual class labels
def computeError(predict, actual):
    N = predict.size
    wrong = 0
    for i in range(N):
        #print(predict[i], actual[i])
        if predict[i] != actual[i]:
            wrong += 1
    return wrong/N

if __name__ == "__main__":
    data = process_data(FILE_NAME)
    N, M = data.shape
    categories = len(CATEGORICAL_COLUMNS)
    
    probabilities = np.zeros((2,))
    feature_data = np.zeros((N))
    
    targets = data.get(TARGET_KEY)
    targets = np.array(pd.DataFrame(targets))
    targets = np.reshape(targets, N)
    
    # For every category, get predictions
    for i in range(categories):
        # Retrieve features and targets from data
        category_data = np.array(data.get(CATEGORICAL_COLUMNS[i]))
        feature_data = np.vstack((feature_data, category_data))
        
        # Format data into numpy arrays
        features = np.array(pd.DataFrame(category_data))
        features = np.reshape(features, N)
        
        # Use Naive-Bayes to find probabilities
        priors, prob = naivebayes(targets[:600], features[:600])
        probabilities = np.vstack((probabilities, prob))
        
    feature_data = np.transpose(feature_data[1:])
    probabilities = np.transpose(probabilities[1:])
    
    # Predict using probabilities
    predictions = predict(probabilities, feature_data[:600])
    error = computeError(predictions, targets[:600])
    print(f'Categorical Probabilities: {probabilities}')
    print(f'Training Error = {error}')
    
    # Testing error
    predictions = predict(probabilities, feature_data[600:])
    error = computeError(predictions, targets[600:])
    print(f'Testing Error = {error}')
    