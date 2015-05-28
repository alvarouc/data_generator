'''
This module contains the data generator class
'''

from smri_gen.rv_gen import mv_rejective
from ica import ica1
import numpy as np
from numpy.random import multivariate_normal as mn


class DataGenerator(object):
    '''
    Class that generates data using ICA and a RV generator method
    '''
    def __init__(self, data, n_components=15):
        '''
        Method that initializes the Data generator. It runs ICA on
        data with n_components
        '''
        # Set number of components to 15,
        # TO DO: use estimation of number of sources from data
        # pre-processing data
        self.data_mean = data.mean(axis=0).reshape((1,-1))
        self.data = data - self.data_mean
        
        # Running ICA
        self.mixing, self.sources\
            = ica1(self.data, n_components, verbose=False)


    def generate(self, n_samples, subset=None, method = 'normal'):

        if method == 'normal':
            sample_mean = np.mean(self.mixing[subset,:], axis=0)
            sample_cov = np.cov(self.mixing[subset,:], rowvar=0)
            new_mixing = mn(sample_mean, sample_cov, n_samples)

        if method == 'rejective':
            model = [np.histogram(column, density=True, bins=20)
                     for column in self.mixing[subset,:].T]
            new_mixing = mv_rejective(model, n_samples)

        new_data = np.dot(new_mixing, self.sources) + self.data_mean
        return new_data

    def generate_by_group(self, n_samples, labels, method='normal'):
        '''
        Generates samples (n_samples) with the specified method
        Input
         -n_samples: number of samples to generate
         -method: 'rejective' or 'normal'
        Output
         -new_samples
        '''
        new_mixings = []
        for tag in np.unique(labels):
            new_mixings.append(self.generate(n_samples, labels==tag, method))

 
        n_groups = np.unique(labels).shape[0]
        new_data = np.zeros((n_samples*n_groups,
                             self.data.shape[1]))
        for idx, mix in enumerate(new_mixings):
            new_data[idx*n_samples:(idx+1)*n_samples, :]\
                = np.dot(mix, self.sources) + self.data_mean

        return new_data

