'''
Unit tests for preprocessor classes in src/preprocessor.py
To run: make test
'''
from numpy import arange, array
from numpy.ma import allequal
from numpy import in1d
import unittest as unittest
import src

def load_txt(filename):
    file_object = file(filename)
    txtstring = file_object.read()
    file_object.close()
    return txtstring

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

    def test_tokenize_vectorizer(self):
        spacy_out_1 = array([[100, 1],
                             [110, 1],
                             [111, 0]])
        spacy_out_2 = array([[100, 1, 1],
                             [110, 1, 0],
                             [111, 0, 0]])
        expected_1 = array([100, 110])
        expected_2 = array([100])

        mat_to_vec = src.preprocessing.doc_tokenizer._vectorize_s_matrix
        actual_1 = mat_to_vec(spacy_out_1)
        actual_2 = mat_to_vec(spacy_out_2)

        self.assertTrue(allequal(actual_1, expected_1))
        self.assertTrue(allequal(actual_2, expected_2))

    def test_spacy_tokenizer(self):
        txt_document = unicode(load_txt('tests/test_document.txt'))
        word_count = 16
        txt_corpus = [txt_document, txt_document]
        
        tokenizer = src.preprocessing.doc_tokenizer(testing = True)
        self.assertTrue(tokenizer.transform(txt_corpus))
        
        result = tokenizer.transform(txt_corpus)
        self.assertEqual(result[0].shape[0], word_count)
        self.assertTrue(allequal(result[0], result[1]))

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

        # Assert fit_transform also works
        filter = src.preprocessing.token_filter(exclude = [], exclude_f = b_func)
        actual = filter.fit_transform(X, y)
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
