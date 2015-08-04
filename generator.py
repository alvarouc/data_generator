'''
This module contains the data generator class
'''
from future.utils import implements_iterator
from .rv_gen import mv_rejective
from ica import ica1
import numpy as np
from scipy.stats import multivariate_normal as mn
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def empirical_mn(mean, covar, size):
    MASS = importr('MASS')
    sample_mean = robjects.FloatVector(mean)
    temp = robjects.FloatVector(covar.ravel())
    sample_cov = robjects.r['matrix'](temp, nrow=covar.shape[0])
    new_mixing = np.array(MASS.mvrnorm(n=size, mu=sample_mean,
                                       Sigma=sample_cov,
                                       empirical=True))
    return new_mixing


@implements_iterator
class DataGeneratorByGroup(object):

    def __init__(self, data, labels,
                 n_components=20,
                 n_samples=100,
                 n_batches=1000,
                 method='normal'):
        self.method = method
        self.n_batches = n_batches
        data =( data[labels==0,:], data[labels==1,:])
        self.data_mean = (x.mean(axis=1).reshape((-1,1)) for x in data)
        model = ica1(n_components)
        (mix1, self.source1), (mix2, self.source2) =\
            (model.fit(x) for x in data)

        self.parameters = {
            'sample_mean':(x.mean(axis=0) for x in [mix1,mix2]),
            'sample_cov':(np.cov(x, rowvar=0) for x in [mix1,mix2]),
            'sample_hist':([np.histogram(column, density=True, bins=20)[0]
                            for column in x.T] for x in [mix1, mix2]),
            'n_samples':n_samples}
        
    def __iter__(self):
        self.batch = 0
        return self

    def __next__(self):

        self.batch +=1
        if self.batch > self.n_batches:
            raise StopIteration
        
        if self.method == 'normal':
            new_mixing0 = empirical_mn(self.parameters['sample_mean'][0],
                             self.parameters['sample_cov'][0],
                             self.parameters['n_samples'])
            new_mixing1 = empirical_mn(self.parameters['sample_mean'][1],
                             self.parameters['sample_cov'][1],
                             self.parameters['n_samples'])

        if self.method == 'rejective':
            new_mixing0 = mv_rejective(self.parameters['sample_hist'][0],
                                      self.parameters['n_samples'])
            new_mixing1 = mv_rejective(self.parameters['sample_hist'][1],
                                      self.parameters['n_samples'])

        new_data0 = np.dot(new_mixing0, self.sources) + self.data_mean[0]
        new_data1 = np.dot(new_mixing1, self.sources) + self.data_mean[1]
        return np.vstack((new_data0,new_data1))
    

    
@implements_iterator
class DataGenerator(object):
    '''
    Class that generates data using ICA and a RV generator method
    '''
    def __init__(self, data, 
                 n_components=20,
                 n_samples=100,
                 n_batches=1000,
                 method='normal'):
        '''
        Method that initializes the Data generator. It runs ICA on
        data with n_components
        '''
        self.method = method
        self.n_batches = n_batches
        #self.data_mean = data.mean(axis=1).reshape((-1,1))
        
        # Running ICA
        model = ica1(n_components)
        mixing, self.sources = model.fit(data)

        self.parameters = {
            'sample_mean':np.mean(mixing, axis=0),
            'sample_cov':np.cov(mixing, rowvar=0),
            'sample_hist':[np.histogram(column, density=True, bins=20)
                           for column in mixing.T],
            'n_samples':n_samples,
        }


    def __iter__(self):
        self.batch=0
        return self
        
    def __next__(self):
        self.batch += 1
        if self.batch > self.n_batches:
            raise StopIteration
        if self.method == 'normal':
            new_mixing = empirical_mn(self.parameters['sample_mean'],
                                      self.parameters['sample_cov'],
                                      self.parameters['n_samples'])

        if self.method == 'rejective':
            new_mixing = mv_rejective(self.parameters['sample_hist'],
                                      self.parameters['n_samples'])

        new_data = np.dot(new_mixing, self.sources)# + self.data_mean
        return new_data

    
