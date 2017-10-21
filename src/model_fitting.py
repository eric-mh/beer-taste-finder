'''
Stores the complete data to model pipeline.
'''

from numpy import array, vstack
import matplotlib.pyplot as plt

def plot_score_dropoff(generic_model):
    plt.plot(generic_model.top_tokens().T[1].astype(float))
    plt.show()

class generic():
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
    def __init__(self, pipeline, X, y, model):
        self.pipeline = pipeline
        self.model = model()
        self.X = X
        self.y = y

        self.not_run = True

    def _run(self):
        """ Run the linear model on the specified data using a specified preprocessor
        and on the specified data. Is required before any other method is called. """
        self.y = array(list(self.y))*10
        self.X = self.pipeline.fit_transform(self.X, self.y)
        self.model.fit(self.X, self.y)
        self.not_run = False

    def predict(self,X):
        """
        INPUTS:
        -------
            X : iter, an interable object containing documents to predict on.
        OUTPUT:
        -------
            y : array, target predictions for every document. """
        if self.not_run:
            self._run()
        X_transformed = self.pipeline.transform(X)
        return self.model.predict(X_transformed)

    def score(self, X = None, y = None):
        """
        INPUTS:
        -------
            X: iter, (optional) an iterable object containing test documents.
            y: iter, (optional) an iterable object containing true targets.
        
        If either X or y are None, .score will score on the training data.

        OUTPUT:
        -------
            score: float, The score of the model on X and y. """
        if self.not_run:
            self._run()
        if X == None or y == None:
            y_transformed = self.y
            X_transformed = self.X
        else:
            y_transformed = array(list(y))
            X_transformed = self.pipeline.transform(X)
        return self.model.score(X_transformed, y_transformed)

    def top_tokens(self):
        """ Returns an array shaped (C, 2) for C distinct tokens in the vocabulary.
        Each pair is ordered word_pair, feature_importance, and is sorted in descending
        importances. """
        importances = self.model.feature_importances_
        vocabulary = self.pipeline.feature_vocabulary_
        return array(sorted(vstack((vocabulary, importances)).T,
                            key = lambda x: float(x[1])))[::-1]

