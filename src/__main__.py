"""
src/main holds different preprocessing / modeling runs.
"""
import ratings_importer
import modeling
import preprocessing
import model_fitting

from model_fitting import plot_score_dropoff
from time import time

def run_nb_pipeline(beer_style = 'English Stout', size = 240):
    "Test to see if naive bayes can be used during model_fitting."
    mongo_gen = ratings_importer.MongoGenerator
    filter_query = {'beer/style' : beer_style}
    feature_key = 'review/text'
    target_key = 'review/taste'

    data_X = mongo_gen(filter_query = filter_query, key = feature_key, limit = size)
    data_y = mongo_gen(filter_query = filter_query, key = target_key, limit = size)

    pipeline = preprocessing.SimplePipeline(
        step_kwargs= [{'batch_size': size/4, 'n_threads': 4, 'testing':True},
                      {'collection':[], 'collect_func': None, 'exclude': True},
                      {'threshold': 0.5, 'metric': None},
                      {'use_tfidfs': True}],
        write_stats = True)

    pipeline_model = model_fitting.generic(pipeline = pipeline,
                                               X = data_X,
                                               y = data_y,
                                               model = modeling.NBImportances)
    t = time()
    pipeline_model._run()
    print("Time to run model: {} s".format(time() - t))
    return pipeline_model

mp = run_nb_pipeline(beer_style = 'Kvass', size = 240)
print mp.top_tokens()[:20]
