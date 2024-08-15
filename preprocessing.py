import h5py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


file_path = './data.h5'
f = h5py.File(file_path, 'r')

keys = list(f.keys())


# Get the keys in the HDF5 file
print("Keys in the HDF5 file:")
print((f['showers'][0].transpose().shape))

# Load your data from the HDF5 file
showers = np.array(f['showers'])

def preprocess_data(data):
    # Reshape to 2D for normalization
    original_shape = data.shape
    reshaped_data = data.reshape(-1, original_shape[-1])
    
    # Normalize each feature (last dimension)
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(reshaped_data)
    
    # Reshape back to the original shape
    normalized_data = normalized_data.reshape(original_shape)
    
    return normalized_data, scaler

normalized_data, scaler = preprocess_data(showers)
