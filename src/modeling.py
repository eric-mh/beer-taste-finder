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

class linear_importances():
    """ A wrapper for sklearn's LinearRegression that calculates feature importances
    as their corresponding beta cofficients divided by the sum of its frequency in
    the data. """
    def __init__(self, **kwargs):
        self.linear_model = LinearRegression(**kwargs)
        self.feature_importances_ = None

    def _calc_importance(self):
        pass

    def fit(self, X, y):
        pass

class nb_importances():
    """ A wrapepr for sklearn's MultinomialNB that calculates feature importances
    as the total impact of that feature in the confidence of the model. """
    def __init__(self, **kwargs):
        self.nb_model = MultinomialNB(**kwargs)
        self.feature_importances_ = None

    def _calc_importance(self):
        pass

    def fit(self, X, y):
        pass
