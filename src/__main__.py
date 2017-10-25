"""
src/main holds different preprocessing / modeling runs.
"""
import ratings_importer
import modeling
import preprocessing
import model_fitting

from time import time

from numpy import logspace

def process_nbe_model(beer_style = 'English Stout', size = 2400):
    """A simplified pipeline with exceptional/not exceptional labels,
    and cutting out the most costly step in the preprocessor in favor of a simple
    tfidfs excluder."""
    mongo_gen = ratings_importer.MongoGenerator
    filter_query = {'beer/style' : beer_style}

def run_nb_pipeline(beer_style = 'English Stout', size = 2400):
    "Test to see if naive bayes can be used during model_fitting."
    mongo_gen = ratings_importer.MongoGenerator
    filter_query = {'style' : beer_style}
    feature_key = 'text'
    target_key = 'taste'

    data_X = mongo_gen(filter_query = filter_query, key = feature_key, limit = size)
    data_y = mongo_gen(filter_query = filter_query, key = target_key, limit = size)

    # Use NB Metric
    metric = preprocessing.TemMetric(use_nb = True)
    pipeline = preprocessing.SimplePipeline(
        step_kwargs= [{'batch_size': size/4, 'n_threads': 4, 'testing':True},
                      {'collection':[], 'collect_func': None, 'exclude': True},
                      {'threshold': 0.5, 'metric': metric},
                      {'use_tfidfs': True}],
        write_stats = True)

    pipeline_model = model_fitting.generic(pipeline = pipeline,
                                           X = data_X,
                                           y = data_y,
                                           model = modeling.NBImportances,
                                           expect_mode = True)
    t = time()
    pipeline_model._run()
    print("Time to run model: {} s".format(time() - t))
    return pipeline_model

def run_nb_tsearch(beer_style = 'English Stout', size = 240):
    mongo_gen = ratings_importer.MongoGenerator
    filter_query = {'style' : beer_style}
    feature_key = 'text'
    target_key = 'taste'

    data_X = list(mongo_gen(filter_query = filter_query, key = feature_key, limit = size))
    data_y = list(mongo_gen(filter_query = filter_query, key = target_key, limit = size))

    metric = preprocessing.TemMetric(use_nb = True)
    score_dict = {}
    for threshold in logspace(-5, 0, 90):
        pipeline = preprocessing.SimplePipeline(
            step_kwargs= [{'batch_size': size/4, 'n_threads': 4, 'testing':True},
                          {'collection':[], 'collect_func': None, 'exclude': True},
                          {'threshold': threshold, 'metric': metric},
                          {'use_tfidfs': True}],
            write_stats = False)

        pipeline_model = model_fitting.generic(pipeline = pipeline,
                                               X = data_X, y = data_y,
                                               model = modeling.NBImportances)
        pipeline_model._run()
        score_dict[threshold] = (pipeline_model.score(classification = True),
                                 len(pipeline_model.top_tokens()))

    return score_dict
