import unittest
from data_generator import (DataGenerator,
                            DataGeneratorByGroup,
                            empirical_mn)
from ica import ica1
from numpy.random import logistic, uniform
from numpy.linalg import norm
from numpy import (eye,
                   histogram,
                   allclose,
                   dot,cov,
                   vstack,
                   corrcoef,
                   abs,sign,
                   array)
import theano.sandbox.cuda
theano.sandbox.cuda.use('cpu')
def synthetic_data():
    mixing_0 = empirical_mn([1,1,1], eye(3), size = 500)
    mixing_1 = empirical_mn([-1,-1,-1], eye(3), size = 500)
    sources = logistic(0,1, size=(3,50000))
    
    data = dot(vstack((mixing_0,mixing_1)), sources) +\
           uniform(low=1, high=2,  size=(1, 50000))
    labels = [0]*500 + [1]*500

    return (data,labels)

class TestSyntheticDataByGroup(unittest.TestCase):

    def setUp(self):
        self.data, self.labels = synthetic_data()

    def test_new_data(self):
        gen = DataGeneratorByGroup(self.data,
                                   self.labels,
                                   n_components=3,
                                   n_samples=500,
                                   n_batches=10,
                                   method='normal')

        for new_data in gen:
            new_labels = [0]*500 + [1]*500
            print(new_data.shape)
            
        
@unittest.skip("demonstrating skipping")
class TestGeneratorMethods(unittest.TestCase):

    def test_mv_normal(self):
        # three dimensional input samples
        old_data, labels = synthetic_data()
        gen = DataGenerator(old_data, n_components=3,
                            n_samples=1000, n_batches=10,
                            method='normal')
        new_data = [x for x in gen]
        # Checking that there is enough samples
        for i in range(10):
            self.assertEqual(new_data[i].shape[0],1000)
        
        old_model = (gen.parameters['sample_mean'],
                     gen.parameters['sample_cov'])
        
        
        model = ica1(3)
        # foreach data batch
        for i in range(10):
            new_A, new_S = model.fit(new_data[i])
            # Reorder components
            c_SS = corrcoef(new_S, gen.sources)[3:,:3]
            source_sim = abs(c_SS).max(axis=1)
            # Check sources are similar
            self.assertTrue(all(source_sim>0.95))
            order = abs(c_SS).argmax(axis=1)
            signs = array([sign(x[order[n]])
                           for n,x in enumerate(c_SS)])
            new_S = new_S[order,:] * signs.reshape((-1,1))
            new_A = new_A[:,order] * signs.reshape((1,-1))

            new_model = (new_A.mean(axis=0),
                         cov(new_A, rowvar=0))
            # Check that resulting new mixing has similar mean and cov
            error = norm(abs(new_model[0] - old_model[0]))/3
            self.assertTrue(error < 0.01)

            error = norm(abs(new_model[1] - old_model[1]).ravel())/9
            self.assertTrue(error < 0.01)

    
    def test_mv_rejection(self):
        # three dimensional input samples
        old_data, labels = synthetic_data()
        gen = DataGenerator(old_data, n_components=3,
                            n_samples=1000, n_batches=10,
                            method='rejective')
        new_data = [x for x in gen]

        for i in range(10):
            self.assertEqual(new_data[i].shape[0],1000)
        
        old_model = gen.parameters['sample_hist']
        
        model = ica1(3)
        # foreach data batch
        for i in range(10):
            new_A, new_S = model.fit(new_data[i])
            # Reorder components
            c_SS = corrcoef(new_S, gen.sources)[3:,:3]
            source_sim = abs(c_SS).max(axis=1)
            # Check sources are similar
            self.assertTrue(all(source_sim>0.95))
            order = abs(c_SS).argmax(axis=1)
            signs = array([sign(x[order[n]])
                           for n,x in enumerate(c_SS)])
            new_S = new_S[order,:] * signs.reshape((-1,1))
            new_A = new_A[:,order] * signs.reshape((1,-1))

            new_model = [histogram(column, density=True, bins=20)\
                         for column in new_A.T]
            # Check that resulting new mixing has similar histogram
            for j in range(len(new_model)):
                sim = abs(corrcoef(new_model[j][0], old_model[j][0]))
                self.assertTrue( sim[0,1] > 0.8,
                                'simmilarity {} is too low'.format(sim))
                


if __name__ == '__main__':

    unittest.main()
