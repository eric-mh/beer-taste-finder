'''
Unit tests for modeling classes in src/modeling.py
To run: make test or make test_modeling
'''
import unittest as unittest
import src

class TestModeling(unittest.TestCase):
    def test_module_exists(self):
        self.assertTrue(src.modeling.linear_importances)
        self.assertTrue(src.modeling.nb_importances)
