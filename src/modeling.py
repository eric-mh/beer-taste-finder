"""
Modeling.py holds models and model wrappers used in the main part of this project:
quantifying the importance of word features in text.

So far, this file contains:
    - linear_importances:
        * Calculate importances from a linear regression. Intended only for testing.
    - nb_importances:
        * Calculate importances from a naive bayes multinomial classifier.
"""
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import r2_score

from numpy import zeros, maximum, minimum, array

class LinearImportances():
    """ A wrapper for sklearn's LinearRegression that calculates feature importances
    as their corresponding beta cofficients divided by the sum of its frequency in
    the data. """
    def __init__(self, **kwargs):
        self.linear_model = LinearRegression(**kwargs)
        self.feature_importances_ = None

    def _calc_importances(self, X, y):
        scores = (self.linear_model.coef_ + self.linear_model.intercept_)
        token_frequencies = (X > 0).sum(axis = 0)
        self.feature_importances_ = scores/token_frequencies
        return self.feature_importances_

    def fit(self, X, y):
        self.linear_model.fit(X, y)
        self._calc_importances(X, y)
        return self

    def predict(self, X):
        return self.linear_model.predict(X)

    def score(self, X, y):
        return self.linear_model.score(X, y)

class NBImportances():
    """ A wrapepr for sklearn's MultinomialNB that calculates feature importances
    as the total impact of that feature in the confidence of the model.

    For now, calculates the positive impact of a feature instead of confidence to
    avoid having to re-train the model for every column."""
    def __init__(self, **kwargs):
        self.sk_model = MultinomialNB(**kwargs)
        self.feature_importances_ = None

    def _calc_importances(self, X, y):
        token_frequencies = (X > 0).sum(axis = 0)

        scores = []
        empty = zeros(X.shape[0])
        for col in range(X.shape[1]):
            swap = X.T[col].copy()
            X.T[col] = empty.copy()
            # Net metric
            scores.append(maximum(y - self.predict(X), empty).sum())
            X.T[col] = swap

        self.feature_importances_ = array(scores)*X.sum(axis = 0)

    def fit(self, X, y):
        self.sk_model.fit(X, y.astype(int))
        self._calc_importances(X, y)
        return self

    def predict(self, X):
        return self.sk_model.predict(X)

    def score_(self, X, y):
        return self.sk_model.score(X, y)

    def score(self, X, y):
        return r2_score(y, self.predict(X), sample_weight=None,
                        multioutput='variance_weighted')
