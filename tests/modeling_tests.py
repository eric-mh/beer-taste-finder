'''
Unit tests for modeling classes in src/modeling.py
To run: make test or make test_modeling
'''
import unittest as unittest
from numpy.ma import allequal
import src

from numpy import array

class TestModeling(unittest.TestCase):
    def test_module_exists(self):
        self.assertTrue(src.modeling.linear_importances)
        self.assertTrue(src.modeling.nb_importances)

    def test_linear_wrapper(self):
        X = array([[1, 0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0, 1],
                   [0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 1, 1]])
        y = array([1., 2., 3., 4., .5])

        linear_wrapper = src.modeling.linear_importances()
        linear_wrapper.fit(X, y)
        self.assertTrue(allequal(y, linear_wrapper.predict(X)))
        self.assertIsNotNone(linear_wrapper.feature_importances_)
