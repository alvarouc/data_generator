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
    def __init__(self, data, labels, n_components=15):
        '''
        Method that initializes the Data generator. It runs ICA on
        data with n_components
        '''
        # Set number of components to 15,
        # TO DO: use estimation of number of sources from data
        self.labels = labels
        # acconditioning data
        self.data_mean = data.mean(axis=0).reshape((1,-1))
        self.data = data - self.data_mean
        
        # Running ICA
        self.mixing, self.sources\
            = ica1(self.data, n_components, verbose=False)
        self.new_mixings = []

    def generate(self, n_samples, method='rejective'):
        '''
        Generates samples (n_samples) with the specified method
        Input
         -n_samples: number of samples to generate
         -method: 'rejective' or 'normal'
        Output
         -new_samples
        '''
        self.new_mixings = []
        if method == 'rejective':
            for tag in np.unique(self.labels):
                model = [np.histogram(column, density=True, bins=20)
                         for column in self.mixing[self.labels == tag, :].T]
                self.new_mixings.append(mv_rejective(model, n_samples))

        elif method == 'normal':
            for tag in np.unique(self.labels):
                sample_mean = np.mean(self.mixing, axis=0)
                sample_cov = np.cov(self.mixing, rowvar=0)
                self.new_mixings.append(mn(sample_mean, sample_cov, n_samples))

        else:
            print 'method: ' + method + ', not implemented'

        n_groups = np.unique(self.labels).shape[0]
        new_data = np.zeros((n_samples*n_groups,
                             self.data.shape[1]))
        for idx, mix in enumerate(self.new_mixings):
            print self.data_mean.shape
            new_data[idx*n_samples:(idx+1)*n_samples, :]\
                = np.dot(mix, self.sources) + self.data_mean

        return new_data

