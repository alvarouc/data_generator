'''
Demo of data generation
'''

from smri_gen.generator import DataGenerator
import numpy as np


def main():
    '''
    Loads data in `data` and `labels` then runs data generator on data
    to generate new data
    '''
    # Prepare data
    path = '/home/aulloa/data/mega/'
    data = np.load(path + 'dataQC.npy')
    labels = np.load(path + 'labelQC.npy').ravel()

    # Initialize data generator with training data
    generator = DataGenerator(data, labels)

    # generating new samples
    n_samples = 5000
    new_data = generator.generate(n_samples, method='normal')
    new_labels = [0]*n_samples + [1]*n_samples
    # Saving data
    np.save(path + 'new_data.npy', new_data)
    np.save(path + 'new_labels.npy', new_labels)

if __name__ == 'main':
    main()



