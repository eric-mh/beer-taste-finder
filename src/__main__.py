"""
src/main holds different preprocessing / modeling runs.
"""
import ratings_importer
import modeling
import preprocessing
import pipeline_modeling

from info_data import info_container

from time import time
from numpy import logspace

generic = pipeline_modeling.generic

def model_top_all(model = modeling.NBExceptional):
    beer_styles = info_container.beer_styles
    exclude_list = info_container.custom_stopwords
    all_tokens = set()
    for beer_style in beer_styles:
        pipeline_model = generic(excludes = exclude_list,
                                 min_df = None, max_df = None,
                                 model = model)
        pipeline_model.fit(X, y)
        top_20 = pipeline_model.top_tokens()[:20].T[0]
        all_tokens.update(top_20)
    return all_tokens
