'''
Unit tests for preprocessor classes in src/preprocessor.py
To run: make test
'''
from numpy import arange, array, ndarray
from numpy.ma import allequal
from numpy import in1d
import unittest as unittest
import src

src.load_preprocessing()

def load_txt(filename):
    file_object = file(filename)
    txtstring = file_object.read()
    file_object.close()
    return txtstring

class TestPreprocessing(unittest.TestCase):
    def assert_equal_array(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertIsNotNone(expected)
        for a_row, e_row in zip(actual, expected):
            self.assertTrue(allequal(a_row, e_row))

    def test_module_exists(self):
        """ Test that everything from preprocessing.py is loaded. """
        self.assertTrue(src.preprocessing.doc_tokenizer)
        self.assertTrue(src.preprocessing.token_filter)
        self.assertTrue(src.preprocessing.token_vectorizer)
        self.assertTrue(src.preprocessing.tem_token_preprocessor)
        self.assertTrue(src.preprocessing.tem_metric)

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
        self.assertTrue(type(result[0]) == ndarray)
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
        filter = src.preprocessing.token_filter(collection = [], collect_func = b_func)
        filter.fit(X, y)

        self.assertTrue(allequal(filter._collection, y + 1))

        # Assert actual and expected arrays have the same elements
        actual = filter.transform(X)
        self.assert_equal_array(actual, expected)

        # Assert fit_transform also works
        filter = src.preprocessing.token_filter(collection = [], collect_func = b_func)
        actual = filter.fit_transform(X, y)
        self.assert_equal_array(actual, expected)

    def test_filter_transform(self):
        X = array([arange(10),
                   arange(10),
                   arange(5,10)])
        b = array([0,6,7,8,9])
        expected = array([arange(1,6), arange(1,6), arange(5,6)])

        filter = src.preprocessing.token_filter(collection = b)

        # Assert actual and expected arrays have the same elements
        actual = filter.transform(X)
        self.assert_equal_array(actual, expected)

    def test_vectorizer(self):
        corpus = array([[1,2,3,4],
                        [1,2,5]])
        new_corpus = array([[1,2,7,8]])
        vectorizer = src.preprocessing.token_vectorizer(use_tfidfs = False)

        expected_counts_o1 = 4
        expected_counts_o2 = 3
        expected_counts_new = 2
        actual_old = vectorizer.fit_transform(corpus)
        actual_new = vectorizer.transform(new_corpus)

        self.assertEqual(actual_old[0].sum(), expected_counts_o1)
        self.assertEqual(actual_old[1].sum(), expected_counts_o2)
        self.assertEqual(actual_new.sum(), expected_counts_new)

    def test_tem_preprocessor(self):
        test_tokens = array([array([1,1,1,1,1,2,2,2,2,2,3,3,3,4]),
                             array([1,2,4,5])])
        test_unseen = array([[1,2,4,5]])
        test_metric = src.preprocessing.tem_metric()
        test_metric._metric = lambda X, y = None: X.sum(axis = 0) >= 3

        expected_new = array([[1,1,1,1,1,2,2,2,2,2,3,3,3],
                              [1,2]])
        expected_unseen = array([[1,2]])
        test_threshold = 0.5

        preprocessor = src.preprocessing.tem_token_preprocessor(test_threshold,
                                                                test_metric)

        actual_new = preprocessor.fit_transform(test_tokens)
        actual_unseen = preprocessor.transform(test_unseen)

        self.assert_equal_array(actual_new, expected_new)
        self.assert_equal_array(actual_unseen, expected_unseen)

    def test_metric_basic(self):
        test_tokens = array([[1,1,1,1,1,2,2,2,2,2,3,3,3,4],
                             [1,2,4,5]])
        test_metric = src.preprocessing.tem_metric(use_nb = False)
        test_metric._metric = lambda X, y = None: X.sum(axis = 0) >= 3

        token_scores = test_metric.score_tokens(test_tokens)
        self.assertEqual(token_scores[1].sum(), 3)

    def test_metric_linear(self):
        X = array([[1,6],
                   [2,6],
                   [3,6],
                   [4,6],
                   [5,6]])
        y = array([1,2,3,4,5])
        
        test_metric = src.preprocessing.tem_metric(use_nb = False)
        # Look out for issues with .toarray not being called correctly
        # Doesn't seem to happen during unit testing.
        token_scores = test_metric.score_tokens(X, y)

        threshold = 1.0
        expected = array([True, True, True, True, True, False])
        actual = token_scores[1] >= threshold
        self.assertTrue(allequal(expected, actual))
