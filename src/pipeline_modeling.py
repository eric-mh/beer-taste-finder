'''
Stores the complete data to model pipeline.
'''

from numpy import array, vstack
import preprocessing
pipeline_const = preprocessing.SimplePipeline

class generic():
    """








    PARAMETERS:
    -----------



    METHODS:
    --------


    """
    def __init__(self, model, excludes = [], min_df = None, max_df = None):
        if max_df == None: max_df = 1.0
        if min_df == None: min_df = 1
        self.pipeline = pipeline_const([{'batch_size' : 800, 'n_threads' : 16},
                                        {'collection' : excludes},
                                        {'use_tfidfs' : True,
                                         'min_df' : min_df, 'max_df' : max_df}],
                                       write_stats = False)
        self.model = model()

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

    def score(self, X = None, y = None, classification = False):
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
        if classification:
            return self.model.score_(X_transformed, y_transformed)
        return self.model.score(X_transformed, y_transformed)

    def top_tokens(self):
        """ Returns an array shaped (C, 2) for C distinct tokens in the vocabulary.
        Each pair is ordered word_pair, feature_importance, and is sorted in descending
        importances. """
        importances = self.model.feature_importances_
        vocabulary = self.pipeline.feature_vocabulary_
        return array(sorted(vstack((vocabulary, importances)).T,
                            key = lambda x: float(x[1])))[::-1]

