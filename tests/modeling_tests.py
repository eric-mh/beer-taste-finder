'''
Unit tests for modeling classes in src/modeling.py
To run: make test or make test_modeling
'''
import unittest as unittest
from numpy.ma import allequal, allclose
import src

src.load_modeling()

from numpy import array

X = array([[1, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 1],
           [0, 0, 1, 0, 0, 1],
           [0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 1]])
y = array([1., 2., 3., 4., 5.])

class TestModeling(unittest.TestCase):
    def test_module_exists(self):
        self.assertTrue(src.modeling.LinearImportances)
        self.assertTrue(src.modeling.NBImportances)

    def test_linear_wrapper(self):
        linear_wrapper = src.modeling.LinearImportances()
        linear_wrapper.fit(X, y)
        self.assertTrue(allclose(y, linear_wrapper.predict(X)))
        self.assertIsNotNone(linear_wrapper.feature_importances_)
        self.assertTrue(linear_wrapper.score(X, y) >= 0)

    def test_nb_wrapper(self):
        nb_wrapper = src.modeling.NBImportances()
        nb_wrapper.fit(X, y)
        self.assertTrue(allclose(y, nb_wrapper.predict(X)))
        self.assertIsNotNone(nb_wrapper.feature_importances_)
        self.assertTrue(nb_wrapper.score(X, y) >= 0)

    def test_classification_score(self):
        nb_wrapper = src.modeling.NBImportances()
        nb_wrapper.fit(X, y)
        self.assertIsNotNone(nb_wrapper.score_(X, y))
        self.assertIsNotNone(nb_wrapper.predict_proba(X))

    def test_nbe_wrapper(self):
        nbe_wrapper = src.modeling.NBExceptional()
        nbe_wrapper.fit(X, y)
        self.assertTrue(nbe_wrapper.score(X, y) >= 0)
        self.assertIsNotNone(nbe_wrapper.predict_proba(X))
        self.assertIsNotNone(nbe_wrapper.feature_importances_)
