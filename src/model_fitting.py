'''
Stores all classes containing data to model pipelines.
'''

import preprocessing
from numpy import array

class linear():
    """ The linear pipeline is a first-pass at producing feature importances
    from the reviews data and a RMSE score for the goodness-of-fit. As a first-pass
    approach, it:
        - Skips out meaningful token filtering during preprocessing.
        - Uses a short-cut to value importances (beta coefficients) instead of
          calculating the effect of the feature on the model itself.
        - Does no clustering of either the beers nor the tokens.

    As a regressor, it uses token counts as features to predict the beer aroma score.
    PARAMETERS:
    -----------
        pipeline : SimplePipeline instance that contains all preprocessing steps.
        X : iter, an iterable object containing documents in unicode.
        y : iter, an iterable object containing targets for each document.
    METHODS:
    --------
        predict(X): Predict targets of documents in X.
        score(X, y): Score model with unseen data.
        top_tokens(): Return a list of tokens in descending feature importances"""
    def __init__(self, pipeline, X, y):
        self.pipeline = pipeline
        self.X = X
        self.y = y

        self.is_run = False

    def _run(self):
        """ Run the linear model on the specified data using a specified preprocessor
        and on the specified data. Is required before any other method is called. """
        pass

    def predict(self,X):
        """
        INPUTS:
        -------
            X : iter, an interable object containing documents to predict on.
        OUTPUT:
        -------
            y : array, target predictions for every document. """
        pass

    def score(self, X, y):
        """
        INPUTS:
        -------
            X: iter, an iterable object containing test documents.
            y: iter, an iterable object containing true targets.
        OUTPUT:
        -------
            score: float, The score of the model on X and y. """
        pass

    def top_tokens(self, ):
        """ Returns an array shaped (C, 2) for C distinct tokens in the vocabulary.
        Each pair is ordered word_pair, feature_importance, and is sorted in descending
        importances. """
        pass
