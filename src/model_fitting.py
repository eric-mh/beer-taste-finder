'''
Stores all classes containing data to model pipelines.
'''

import preprocessing

class linear_pipeline():
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
        cross_val : Int, number of folds to use during cross-validation
        beer_type : string, type of beer to use. Not optional because the dataset
                    is too large to run on without meaningful feature space reduction.
    ATTRIBUTES:
    -----------
        importances_ : Array, (2, C) for C distinct tokens,
                       An array with token - score pairings.
        cv_score_ : Float, a cross-val score of the model. """
