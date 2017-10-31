"""
src/main holds different preprocessing / modeling runs.
"""
import modeling
import preprocessing
import pipeline_modeling

from mongo_interface import mongo_loader
from info_data import info_container

from time import time
from numpy import logspace

cv_scorer = pipeline_modeling.cv_scorer
generic = pipeline_modeling.generic

default_style = 'American Double / Imperial IPA'
def cv_scan(beer_style = default_style, model = modeling.NBExceptional, limit = 10000):
    exclude_list = info_container.custom_stopwords
    X, y = mongo_loader(key = 'style', filter = beer_style, limit = limit)
    pipeline_model = generic(excludes = exclude_list,
                             min_df = None, max_df = None,
                             model = model)
    std, mean = 1, 0
    for i in range(2,150):
        if std < abs(0.05*mean):
            return std, mean, i
        cv_scores = cv_scorer(pipeline_model, X, y)
        std, mean = cv_scores.std(), cv_scores.mean()
        print std, mean, i

def model_top_one(beer_style = default_style, model = modeling.NBExceptional, limit = None):
    exclude_list = info_container.custom_stopwords
    X, y = mongo_loader(key = 'style', filter = beer_style, limit = limit)
    pipeline_model = generic(excludes = exclude_list,
                             min_df = None, max_df = None,
                             model = model)
    pipeline_model.fit(X, y)
    print pipeline_model.top_tokens()[:20]
    return pipeline_model, X, y

def model_top_all(model = modeling.NBExceptional, limit = None):
    beer_styles = info_container.beer_styles
    exclude_list = info_container.custom_stopwords
    all_tokens = set()
    for beer_style in beer_styles:
	X, y = mongo_loader(key = 'style', filter = beer_style, limit = limit)
        pipeline_model = generic(excludes = exclude_list,
                                 min_df = None, max_df = None,
                                 model = model)
        pipeline_model.fit(X, y)
        top_20 = pipeline_model.top_tokens()[:20].T[0]
	all_tokens.update(top_20)
    return all_tokens
