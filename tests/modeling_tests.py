'''
Unit tests for modeling classes in src/modeling.py
To run: make test or make test_modeling
'''
import unittest as unittest
from numpy.ma import allequal, allclose
import src

src.load_modeling()

from numpy import array

class TestModeling(unittest.TestCase):
    def test_module_exists(self):
        self.assertTrue(src.modeling.LinearImportances)
        self.assertTrue(src.modeling.NBImportances)

    def test_linear_wrapper(self):
        X = array([[1, 0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0, 1],
                   [0, 0, 1, 0, 0, 1],
                   [0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 1, 1]])
        y = array([1., 2., 3., 4., 5.])

        linear_wrapper = src.modeling.LinearImportances()
        linear_wrapper.fit(X, y)
        self.assertTrue(allclose(y, linear_wrapper.predict(X)))
        self.assertIsNotNone(linear_wrapper.feature_importances_)
        self.assertTrue(linear_wrapper.score(X, y) >= 0)
