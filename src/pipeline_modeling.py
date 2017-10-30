'''
Stores the complete data to model pipeline.
'''

import inspect

from sklearn.model_selection import cross_val_score, KFold
from numpy import array, vstack
import preprocessing
pipeline_const = preprocessing.SimplePipeline

def cv_scorer(estimator, X, y, cv = 3):
    """ A simpler copy of sklearn's cross_val_score that runs faster.
    Intended to be passed a generic estimator instance defined below. """
    scores = []
    y = y.astype(float)
    kf = KFold(n_splits = cv)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[[train_index]], X[[test_index]]
        y_train, y_test = y[[train_index]], y[[test_index]]
        estimator.fit(X_train, y_train)
        scores.append(estimator.score(X_test, y_test))
    return array(scores)

class generic():
    """
    A rebuilt generic estimator class that can wrap a pipeline and a model.
    Is reduced to only work with a SimplePipeline with 3 preprocessor steps;
    parsing, stripping and vectorization, and a Naive Bayes model from src.modeling.
    PARAMETERS:
    -----------
        model : class object from src.modeling
        excludes : A list of tokens to exclude during preprocessing
        min_df : min_df argument to be passed to the vectorizer
        max_df : max_df argument to be passed to the vectorizer

    METHODS:
    --------
        top_tokens : returns a list of top tokens as defined by the NB model. """
    def __init__(self, model, excludes = [], min_df = None, max_df = None):
        if max_df == None: max_df = 1.0
        if min_df == None: min_df = 1
        self.pipeline = pipeline_const([{'batch_size' : 800, 'n_threads' : 16},
                                        {'collection' : excludes},
                                        {'use_tfidfs' : True,
                                         'min_df' : min_df, 'max_df' : max_df}],
                                       write_stats = False)
        if inspect.isclass(model):
            self._model = model()
        else:
            self._model = model

    def fit(self, X, y):
        """ Mirrors sklearn estimator fit method. """
        X_transformed = self.pipeline.fit_transform(X, y)
        self._model.fit(X_transformed, y.astype(float))

    def predict(self,X):
        """ Mirrors sklearn estimator predict method. """
        X_transformed = self.pipeline.transform(X)
        return self._model.predict(X_transformed)

    def score(self, X, y):
        """ Mirrors sklearn estimator score method. """
        X_transformed = self.pipeline.transform(X)
        return self._model.score(X_transformed, y.astype(float))

    def get_params(self, deep = None):
        """ A minimal mirror of a sklearn estimator's get_params method. """
        return {'model' : self._model}

    def top_tokens(self):
        """ Returns an array shaped (C, 2) for C distinct tokens in the vocabulary.
        Each pair is ordered word_pair, feature_importance, and is sorted in descending
        importances. """
        importances = self._model.feature_importances_
        vocabulary = self.pipeline.feature_vocabulary_
        return array(sorted(vstack((vocabulary, importances)).T,
                            key = lambda x: float(x[1])))[::-1]

