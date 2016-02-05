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
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    def __init__(self, X, y, n_components=20, n_samples=100,
                 n_batches=1000, method='normal',
                 decomposition_method='ica'):

        self.decomposition_method = decomposition_method
        self.n_components = n_components
        self.n_samples = n_samples
        self.method = method
        self.n_batches = n_batches

        assert len(X) == len(y)

        if decomposition_method == 'ica':
            model = ica(n_components)
            self.mixing, self.sources = model.fit(X)
        elif decomposition_method == 'pca':
            model = PCA(n_components)
            self.mixing = model.fit_transform(X)
            self.sources = model.components_
        elif decomposition_method == 'sparsePCA':
            model = SparsePCA(n_components, alpha=0.01)
            self.mixing = model.fit_transform(X)
            self.sources = model.components_
        else:
            logger.info('Method: {}, not implemented'.format(
                decomposition_method))

        # Encode labels
        self.le = LabelEncoder()
        self.le.fit(y)
        logger.info('Classes: {}'.format(self.le.classes_))
        self.y = self.le.transform(y)
        self.n_classes = len(self.le.classes_)

        # partition mixing matrix by label
        self.params = {'mean': [], 'cov': [], 'hist': []}
        for label in range(self.n_classes):
            keep = np.where(self.y == label)
            mix = self.mixing[keep]
            if method == 'normal':
                self.params['mean'].append(mix.mean(axis=0))
                self.params['cov'].append(np.cov(mix, rowvar=0))
            elif method == 'rejective':
                self.params['hist'].append(
                    [np.histogram(column, density=True, bins=20)
                     for column in mix.T])
            else:
                logger.info('Method {}, not implemented'.format(method))
        self.batch = 0

    @property
    def batch_label(self):

        labels = []
        for label in range(self.n_classes):
            true_label = self.le.inverse_transform(label)
            labels.extend([true_label] * self.n_samples)
        return labels

    def __iter__(self):
        self.batch = 0
        return self

    def __next__(self):
        self.batch += 1
        if self.batch > self.n_batches:
            raise StopIteration

        new_data = []
        labels = []
        for aclass in range(self.n_classes):
            if self.method == 'normal':
                new_mix = empirical_mn(self.params['mean'][aclass],
                                       self.params['cov'][aclass],
                                       self.n_samples)
            elif self.method == 'rejective':
                new_mix = mv_rejective(self.params['hist'][aclass],
                                       self.n_samples)
            new_data.append(np.dot(new_mix, self.sources))

            labels = self.batch_label

        return (np.vstack(new_data), labels)


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
