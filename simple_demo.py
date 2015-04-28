'''
Demo of data generation
'''

# package modules
from generator import DataGenerator
# classifiers
import numpy as np

# Prepare data
path = '/home/aulloa/data/mega/'
data = np.load(path + 'dataQC.npy')
labels = np.load(path + 'labelQC.npy').ravel()

# Initialize data generator with training data
generator = DataGenerator(data, labels)

# generating new samples
n_samples = 5000
new_data= generator.generate(n_samples, method='normal')
new_labels = [0]*n_samples + [1]n_samples
# Saving data
np.save(path + 'new_data.npy', new_data)
np.save(path + 'new_labels.npy', new_labels)



