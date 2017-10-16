'''
Unit tests for preprocessor classes in src/preprocessor.py
To run: make test
'''
from numpy import arange, array
from numpy.ma import allequal
from numpy import in1d
import unittest as unittest
import src

class TestPreprocessing(unittest.TestCase):
    def assert_equal_array(self, actual, expected):
        for a_row, e_row in zip(actual, expected):
            self.assertTrue(allequal(a_row, e_row))

    def test_module_exists(self):
        """ Test that everything from preprocessing.py is loaded. """
        self.assertTrue(src.preprocessing.doc_tokenizer)
        self.assertTrue(src.preprocessing.token_filter)
        self.assertTrue(src.preprocessing.token_vectorizer)
        self.assertTrue(src.preprocessing.mfe_metric)

    @unittest.skip("preprocessor pipeline incomplete")
    def test_pipeline(self):
        pass

    @unittest.skip("tokenizer unimplemented")
    def test_tokenizer(self):
        pass

    def test_filter_fit(self):
        X = array([arange(10), arange(13)])
        y = arange(10, step = 2)

        # Assert new X only has items that are in y
        expected = array([array([0, 2, 4, 6, 8]),
                          array([0, 2, 4, 6, 8, 10, 11, 12])])
        b_func = lambda X, y: y + 1

        # Test fitting
        filter = src.preprocessing.token_filter(exclude = [], exclude_f = b_func)
        filter.fit(X, y)

        self.assertTrue(allequal(filter._exclude, y + 1))

        # Assert actual and expected arrays have the same elements
        actual = filter.transform(X)
        self.assert_equal_array(actual, expected)

    def test_filter_transform(self):
        X = array([arange(10),
                   arange(10),
                   arange(5,10)])
        b = array([0,6,7,8,9])
        expected = array([arange(1,6), arange(1,6), arange(5,6)])

        filter = src.preprocessing.token_filter(exclude = b)

        # Assert actual and expected arrays have the same elements
        actual = filter.transform(X)
        self.assert_equal_array(actual, expected)

    @unittest.skip("vectorizer unimplemented")
    def test_vectorizer(self):
        pass

    @unittest.skip("metric function unimplemented")
    def test_metric(self):
        pass
