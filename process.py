import pandas as pd  # Pandas is only used to read the data from
                     # the .csv file
import numpy as np

# Processes the data from the file
def process_data(filename):
    data_set = pd.read_csv(filename)
    data_frames = pd.DataFrame(data_set)
    data = np.array(data_frames)
    
    formatted_data = get_categories(data_set)
    
    # first = data[0]
    # for i in range(len(first)):
    #     d = first[i]
    #     dtype = type(d)
    #     print(f'Data type of {d} is {dtype}')
    
    return formatted_data
        
# TODO: Do get_dummies from scratch
def get_categories(data):
    dummies = pd.get_dummies(data)
    return dummies

def extract_data(df:pd.DataFrame, target_key):
    target = np.array(df.get(target_key))
    features = np.array(df.loc[:, df.columns!=target_key])
    return target, features

def shuffle_and_split(data):
    shuffled = np.random.shuffle(data)
    print(shuffled)