import unittest
from data_generator import mv_rejective, DataGenerator
from ica import ica1
from numpy.random import multivariate_normal as mn
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
    mixing_0 = mn([1,1,1], cov=eye(3), size = 500)
    mixing_1 = mn([-1,-1,-1], cov=eye(3), size = 500)
    sources = logistic(0,1, size=(3,50000))
    
    data = dot(vstack((mixing_0,mixing_1)), sources) +\
           uniform(low=1, high=2,  size=(1, 50000))
    labels = [0]*500 + [1]*500

    return (data,labels)

class TestRVMethods(unittest.TestCase):

    def test_mv_normal(self):
        # three dimensional input samples
        old_data, labels = synthetic_data()
        gen = DataGenerator(old_data, n_components=3,
                            n_samples=1000, n_batches=10,
                            method='normal')
        print('Generating new data ...')
        new_data = [x for x in gen]
        print('Done.')
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
            print('Cov error {}'.format(error))
            self.assertTrue(error < 0.01)
            

        self.fail('finish the test')
    
    def test_mv_rejection(self):
        # three dimensional input samples
        old_data, labels = synthetic_data()
        gen = DataGenerator(old_data, n_components=3,
                            n_samples=1000, n_batches=10,
                            method='rejective')
        print('Generating new data')
        new_data = [x for x in gen]
        print('Done.')

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
                

#class TestGenerator(unittest.TestCase):       
#    def test_data_generator_ica(self):
#
#        data, labels = synthetic_data()
#        gen = DataGenerator(data, labels, n_components=3)
#        # Test if ica decomposition is ok
#        ica_data = dot(gen.mixing, gen.sources) + gen.data_mean
#        self.assertTrue(allclose(ica_data, data))
#
#    def test_data_generator_normal(self):
#        data, labels = synthetic_data()
#        gen = DataGenerator(data, labels, n_components=3)
#        # Test generator
#        new_data = gen.generate(1000, method='normal')
#        self.assertEqual(new_data.shape, (2000,50000))
#        
#    def test_data_generator_rejective(self):
#        data, labels = synthetic_data()
#        gen = DataGenerator(data, labels, n_components=3)
#        # Test generator
#        new_data = gen.generate(1000, method='rejective')
 #       self.assertEqual(new_data.shape, (2000,50000))
        

if __name__ == '__main__':
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRVMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
