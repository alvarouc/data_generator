import unittest
from smri_gen import mv_rejective


class TestRVMethods(unittest.TestCase):
    def test_mv_rejective(self):

        from numpy.random import multivariate_normal as mn
        from numpy import eye, histogram, allclose
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

if __name__ == '__main__':
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRVMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
