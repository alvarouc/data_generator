import unittest
from smri_gen import mv_rejective, DataGenerator
from numpy.random import multivariate_normal as mn
from numpy.random import logistic, uniform
from numpy import eye, histogram, allclose, dot, vstack

def synthetic_data():
    mixing_0 = mn([1,1,1], cov=eye(3), size = 500)
    mixing_1 = mn([-1,-1,-1], cov=eye(3), size = 500)
    sources = logistic(0,1, size=(3,50000))
    
    data = dot(vstack((mixing_0,mixing_1)), sources) +\
           uniform(low=1, high=2,  size=(1, 50000))
    labels = [0]*500 + [1]*500

    return (data,labels)

class TestRVMethods(unittest.TestCase):
    def test_mv_rejective(self):
        # three dimensional input samples
        input_samples = mn(mean=[0,0,0], cov=eye(3), size=10000)
        models = [histogram(column, density=True, bins=20)\
                  for column in input_samples.T]
        out_samples = mv_rejective(models, 5000)
        # Check for output size
        self.assertEqual(out_samples.shape, (5000,3))
        out_models = [histogram(column, density=True, bins=20)\
                      for column in out_samples.T]
        # Check for relative error: allow 1% missmatch
        self.assertTrue(all([allclose(model[0], out_model[0], atol=0.01*20)\
                             for model,out_model in zip(models,out_models)]))

    def test_data_generator_ica(self):

        data, labels = synthetic_data()
        gen = DataGenerator(data, n_components=3)
        # Test if ica decomposition is ok
        ica_data = dot(gen.mixing, gen.sources) + gen.data_mean
        self.assertTrue(allclose(ica_data, data))

    def test_data_generator_normal(self):
        data, labels = synthetic_data()
        gen = DataGenerator(data, n_components=3)
        # Test generator
        new_data = gen.generate(labels, 5000, method='normal')
        self.assertEqual(new_data.shape, (10000,50000))
        
    def test_data_generator_rejective(self):
        data, labels = synthetic_data()
        gen = DataGenerator(data, n_components=3)
        # Test generator
        new_data = gen.generate(labels, 5000, method='rejective')
        self.assertEqual(new_data.shape, (10000,50000))
        

if __name__ == '__main__':
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRVMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
