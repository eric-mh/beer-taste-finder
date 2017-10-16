"""
The preprocessor file holds classes that follow the sklearn convention for preprocessing
the beer data in this project.

Every preprocessor class mimics the self.fit, self.fit_transform and self.transform
behavior as if they adhered to sklearn conventions.
"""
from numpy import array, in1d
# from spacy import load
# parser = load('en_core_web_md')

class doc_tokenizer():
    """ The document tokenizer takes a corpus and tokenizes every document into
    a numpy list of all the words as spacy word IDs. """
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self.transform(X)

    def transform(self, X):
        """ Tokenize all documents.
        INPUTS:
        -------
            X : An interable corpus of documents. For now, expect a generator.
        OUTPUTS:
        --------
            X : The transformed corpus. """
        pass # wt

    def fit_transform(self, X, y = None):
        return self.transform(X)

class token_filter():
    """ _token_filter removes everything inside of self.exclude. Can be initialized
    with a starting set and a function to expand the list.
    PARAMETERS:
    -----------
        exclude : List, optional
            A list of tokens to exclude from the start
        exclude_f : Lambda Function, optional
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
        pass # wt

    def transform(self, X):
        """ Transforms a matrix of tokenized documents into a reduced set.
        INPUTS:
        -------
            X : Array, An np array of tokenized documents
        OUTPUTS:
        --------
            X : Array, A reduced array of tokenized document.
        """
        collector = []
        for row in X:
            collector.append(self._filter_row(row))
        return array(collector)

    def fit_transform(self, X, y = None):
        pass

class token_vectorizer():
    """ DOC """ # DOC
    def __init__(self):
        pass

class mfe_metric():
    """ DOC """ # DOC
    def __init__(self):
        pass
