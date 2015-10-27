'''
This module contains the data generator class
'''
# from future.utils import implements_iterator
from .rv_gen import mv_rejective
from ica import ica
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
#from scipy.stats import ttest_ind
from sklearn.decomposition import PCA, SparsePCA


def empirical_mn(mean, covar, size):
    return np.random.multivariate_normal(mean, covar, size)

def empirical_mnr(mean, covar, size):
    MASS = importr('MASS')
    sample_mean = robjects.FloatVector(np.array(mean))
    temp = robjects.FloatVector(covar.ravel())
    sample_cov = robjects.r['matrix'](temp, nrow=covar.shape[0])
    new_mixing = np.array(MASS.mvrnorm(n=size, mu=sample_mean,
                                       Sigma=sample_cov,
                                       empirical=True))
    return new_mixing


# @implements_iterator
class DataGeneratorByGroup(object):

    def __init__(self, data, labels,
                 n_components=20,
                 n_samples=100,
                 n_batches=1000,
                 method='normal',
                 decomposition_method='ica'):
        self.method = method
        self.n_batches = n_batches
        if decomposition_method == 'ica':
            model = ica(n_components)
            self.mixing, self.sources = model.fit(data)
        if decomposition_method == 'pca':
            model = PCA(n_components)
            self.mixing = model.fit_transform(data)
            self.sources = model.components_
        if decomposition_method == 'sparsePCA':
            model = SparsePCA(n_components, alpha=0.01)
            self.mixing = model.fit_transform(data)
            self.sources = model.components_

        #result = ttest_ind(mixing[labels == 0, :],
        #                   mixing[labels == 1, :])
        #mixing = mixing[:, result.pvalue < 0.01]
        #self.sources = self.sources[result.pvalue < 0.01, :]

        a0 = self.mixing[np.array(labels) == 0, :]
        a1 = self.mixing[np.array(labels) == 1, :]
        self.parameters = {
            'sample_mean': [a0.mean(axis=0), a1.mean(axis=0)],
            'sample_cov': [np.cov(x, rowvar=0) for x in [a0, a1]],
            'sample_hist': [[np.histogram(column, density=True, bins=20)
                            for column in x.T] for x in [a0, a1]],
            'n_samples': n_samples}
        self.batch = 0

    @property
    def batch_label(self):
        n = self.parameters['n_samples']
        return(np.array([0]*n + [1]*n))

    def __iter__(self):
        self.batch = 0
        return self

    def __next__(self):
        self.batch += 1
        if self.batch > self.n_batches:
            raise StopIteration

        if self.method == 'normal':
            new_mix0 = empirical_mn(
                self.parameters['sample_mean'][0],
                self.parameters['sample_cov'][0],
                self.parameters['n_samples'])
            new_mix1 = empirical_mn(
                self.parameters['sample_mean'][1],
                self.parameters['sample_cov'][1],
                self.parameters['n_samples'])

        if self.method == 'rejective':
            new_mix0 = mv_rejective(
                self.parameters['sample_hist'][0],
                self.parameters['n_samples'])
            new_mix1 = mv_rejective(
                self.parameters['sample_hist'][1],
                self.parameters['n_samples'])

        new_data0 = np.dot(new_mix0, self.sources)
        new_data1 = np.dot(new_mix1, self.sources)
        return np.vstack((new_data0, new_data1))


# @implements_iterator
#class DataGenerator(object):
#    '''
#    Class that generates data using ICA and a RV generator method
#    '''
#    def __init__(self, data,
#                 n_components=20,
#                 n_samples=100,
#                 n_batches=1000,#
#
#    def __next__(self):
#        self.batch += 1
#        if self.batch > self.n_batches:
#            raise StopIteration
#        if self.method == 'normal':
#            new_mixing = empirical_mn(self.parameters['sample_mean'],
#                                      self.parameters['sample_cov'],
#                                      self.parameters['n_samples'])#

#        if self.method == 'rejective':
#            new_mixing = mv_rejective(self.parameters['sample_hist'],
#                                      self.parameters['n_samples'])#

#        new_data = np.dot(new_mixing, self.sources)  # + self.data_mean
#        return new_data
