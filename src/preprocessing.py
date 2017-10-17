"""
The preprocessor file holds classes that follow the sklearn convention for preprocessing
the beer data in this project.

Every preprocessor class mimics the self.fit, self.fit_transform and self.transform
behavior as if they adhered to sklearn conventions.
"""
from numpy import array, hstack, min, in1d
from numpy.random import randint
from spacy import load, attrs
from sklearn.feature_extraction import text


class doc_tokenizer():
    """ The document tokenizer takes a corpus and tokenizes every document into
    a numpy list of all the words as spacy word IDs.
    PARAMETERS:
    -----------
        batch_size: int, optional; batch size to pass to parser.pipe
        n_threads: int, optional; number of threads to pass to parser.pipe
        testing: boolean, optional; flag if tokenizer is only being used in a test. """
    def __init__(self, batch_size = 1, n_threads = 1, testing = False):
        if testing:
            self._parser = load('en')
        else:
            self._parser = load('en_core_web_md')

        self.ar_args = [attrs.LEMMA, attrs.IS_ALPHA]

        self._batch_size = batch_size
        self._n_threads = n_threads

    @staticmethod
    def _vectorize_s_matrix(matrix):
        """ Vectorize a single spacy matrix of N x M shape returning the first
        column of N masked with the other columns.
        INPUTS:
        -------
            matrix : N x M numpy array
        OUTPUTS:
        --------
            vector : N, shape numpy array. """
        return matrix[min(matrix[:,1::], axis = 1) == 1][:,0]

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        """ Tokenize all documents. Expects unicode.
        INPUTS:
        -------
            X : An interable corpus of documents. For now, expect a generator.
        OUTPUTS:
        --------
            X : The transformed corpus. """
        collector = [] # Inefficient
        corpus = self._parser.pipe(X, self._batch_size, self._n_threads)

        for document in corpus:
            collector.append(self._vectorize_s_matrix(document.to_array(self.ar_args)))

        return collector

    def fit_transform(self, X, y = None):
        return self.transform(X)

class token_filter():
    """ _token_filter removes everything inside of self.exclude. Can be initialized
    with a starting set and a function to expand the list.
    PARAMETERS:
    -----------
        exclude : List, optional
            A list of tokens to exclude from the start
        exclude_f : Lambda Function F(X, y), optional
            A function that is called during self.fit to expand the exclusion set. """

    def __init__(self, exclude = [], exclude_f = None):
        self._exclude = array(exclude)
        self._exclude_f = exclude_f

    def _filter_row(self, row):
        """ Filters all items not in self.include out of a row
            NOTE - Change the mask to switch include/exclude behavior.
            INPUTS:
            -------
                row : a (N,) shape np array
            OUTPUTS:
            --------
                row : a np row with specific items filtered out
        """
        return row[~in1d(row, self._exclude)]

    def fit(self, X, y = None):
        """ Update the exclusion list with the included function, if there is one. """
        if self._exclude_f:
            self._exclude = hstack((self._exclude, self._exclude_f(X, y)))
        return self

    def transform(self, X):
        """ Transforms a matrix of tokenized documents into a reduced set.
        INPUTS:
        -------
            X : Array, An np array of tokenized documents
        OUTPUTS:
        --------
            X : Array, A reduced array of tokenized document.
        """
        collector = [] # Inefficient
        for row in X:
            collector.append(self._filter_row(row))
        return array(collector)

    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)

class token_vectorizer():
    """ A wrapper for sklearn's text vectorizers that skip the built-in
    preprocessing and tokenization steps.
    PARAMETERS:
    -----------
        use_tfidfs: boolean, optional; use TfidfVectorizer if true. """
    def __init__(self, use_tfidfs = False):
        override = lambda x: x
        if use_tfidfs:
            self.vec = text.TfidfVectorizer(preprocessor = override,
                                            tokenizer = override)
        else:
            self.vec = text.CountVectorizer(preprocessor = override,
                                            tokenizer = override)

    def fit(self, X, y = None):
        """ Fit the vectorizer to a document. """
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        """ Transform the document to a vector matrix. Will ignore any
        volcabulary not seen during fitting.
        INPUTS:
        -------
            X : the corpus; a list of documents where each is a list of tokens.
        OUTPUTS:
        --------
            X, transformed: A sparse matrix with counts or TFIDFS. """
        return self.vec.transform(X)

    def fit_transform(self, X, y = None):
        return self.vec.fit_transform(X, y)

class mfe_preprocessor():
    """ A preprocessor that further reduces the number of features by removing
    'Inefficient' ones. Calculates efficiency using the count vectorizer and a metric.
    PARAMETERS:
    -----------
        threshold : float, a cut-off threshold for the metric.
        metric: Lambda Function(X, y), optional; a function that calculates a metric
                for every feature. Expects a list output that is the positional metric
                of every column. Defaults to a useless metric for testing."""
    def __init__(self, threshold, metric = None):
        self.vectorizer = token_vectorizer(use_tfidfs = False)
        if metric:
            self.metric = metric
        else:
            self.metric = mfe_metric
        self.threshold = threshold

    def fit(self, X, y = None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, y = None):
        pass

def mfe_metric(X, y = None):
    pass
