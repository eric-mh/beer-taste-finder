"""
The preprocessor file holds classes that follow the sklearn convention for preprocessing
the beer data in this project.

Every preprocessor class mimics the self.fit, self.fit_transform and self.transform
behavior as if they adhered to sklearn conventions.

The preprocessing steps are expected to run in this order:
    - preprocessor.doc_tokenizer
        * tokenizes documents with spacy
    - preprocessor.token_filter
        * filters out obvious insignificant tokens 
    - preprocessor.mfe_token_preprocessor
        * further filter out tokens using a metric
    - preprocessor.token_vectorizer
        * converts document tokens into a feature matrix for modeling.
"""
from numpy import array, hstack, vstack, min, in1d
from spacy import load, attrs
from sklearn.feature_extraction import text

from src.modeling import linear_importances, nb_importances

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
    """ token_filter removes everything inside of self._collection. Can be initialized
    with a starting set and a function to expand the list.
    PARAMETERS:
    -----------
        collection : List, optional
            A list of tokens to exclude from the start
        collect_func : Lambda Function F(X, y), optional
            A function that is called during self.fit to expand the collection.
        exclude : boolean, optional
            Specify if token_filter should exclude items in the collection. """

    def __init__(self, collection = [], collect_func = None, exclude = True):
        self._collection = array(collection)
        self._collect_func = collect_func

        if exclude:
            self._filter = lambda row: row[~in1d(row, self._collection)]
        else:
            self._filter = lambda row: row[in1d(row, self._collection)]

    def fit(self, X, y = None):
        """ Update the exclusion list with the included function, if there is one. """
        if self._collect_func:
            self._collection = hstack((self._collection,
                                       self._collect_func(X, y)))
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
            collector.append(self._filter(row))
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
        return self.vec.transform(X).toarray()

    def fit_transform(self, X, y = None):
        return self.vec.fit_transform(X, y).toarray()

class tem_token_preprocessor():
    """ A preprocessor that further reduces the number of tokens by removing
    'Inefficient' ones. Calculates efficiency using a metric function.
    PARAMETERS:
    -----------
        threshold : float, a cut-off threshold for the metric.
        metric: mfe_metric instance, optional; a function that calculates a metric
                for every feature. Expects a list output that is the positional metric
                of every column. Defaults to the model feature efficiency metric."""
    def __init__(self, threshold, metric = None):
        self.vectorizer = token_vectorizer(use_tfidfs = False)
        if metric:
            self.metric = metric
        else:
            self.metric = tem_metric()
        self.threshold = threshold
        self.metric_msk = None

        # Set something aside for the token filter later
        self.excluder = None

    def fit(self, X, y = None):
        token_scores = self.metric.score_tokens(X, y)
        token_mask = token_scores[1] >= self.threshold
        self.excluder = token_filter(collection = token_scores[0][token_mask],
                                     exclude = False)

    def transform(self, X):
        return self.excluder.transform(X)

    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)

class tem_metric():
    """ The model efficiency metric, calculates the 'scores' of each individual token
    when given a corpus of tokens. 
    PARAMETERS:
    -----------
        use_nb : boolean, optional, use nb_importances from src.modeling.
                 Uses src.modeling.linear_importances by default for testing. """
    def __init__(self, use_nb = False):
        self.vectorizer = token_vectorizer(use_tfidfs = False)
        if use_nb:
            self.model = nb_importances()
        else:
            self.model = linear_importances()

    def _metric(self, X, y = None):
        self.model.fit(X, y)
        return self.model.feature_importances_

    def score_tokens(self, X, y = None):
        """ Creates a vector with the scores for each token.
            INPUTS:
            -------
                X : Array, an array of the tokens of N documents, where the total
                    number of distinct tokens is C.
                y : Array, optional, a (N,) shaped np array of targets.
            OUTPUTS:
            --------
                scores : Array shaped (2,C) with token and token scores. """
        scores = self._metric(self.vectorizer.fit_transform(X), y)

        vocab = self.vectorizer.vec.vocabulary_
        vocab = array(sorted(vocab.keys(), key = lambda x: vocab[x]))
        return vstack((vocab, scores))

