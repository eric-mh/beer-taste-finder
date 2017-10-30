'''
Unit tests for the entire pipeline.
Also tests individual model_fitting approaches in src/model_fitting.py
Make test_pipeline to run
'''
from numpy import array, unicode_
import unittest
import src

src.load_pipeline_model()
src.load_modeling()
src.load_mongo_interface()

class TestPipeline(unittest.TestCase):
    X, y = src.mongo_interface.mongo_loader(limit = 80)
    models = {'linear' : LinearImportances,
              'NBI' : NBImportances,
              'NBE' : NBExceptional}
    generic = src.pipeline_model.generic

    @unittest.skip("re-writing tests")
    def test_loaded(self):
        "All required modules are loaded"

    @unittest.skip("re-writing tests")
    def test_NBE_preprocessing(self):
        "Complete test of NBE preprocessing. "
        pipeline_model = generic(
        pipeline_model = src.pipeline_model.generic(pipeline = pipeline,
                                                   X = data_X,
                                                   y = data_y,
                                                   model = src.modeling.NBExceptional)
        pipeline_model._run()

        # Same assertions as in the linear test
        train_score = pipeline_model.score()
        self.assertTrue(train_score != None and train_score != 0)
        top_10 = pipeline_model.top_tokens()[:10].T[0]
        for token in top_10:
            self.assertIn(type(token), [unicode_, unicode])

    @unittest.skip("prioritize NBE")
    def test_preprocessing_pipe(self):
        "Mongo to vector matrix test"
        mongo_gen = src.ratings_importer.MongoGenerator

        filter_query = {'style' : 'Rauchbier'}
        feature_key = 'text'
        target_key = 'taste'

        data_fit_X = mongo_gen(filter_query = None, key = feature_key, limit = 10)
        data_fit_y = mongo_gen(filter_query = None, key = target_key, limit = 10)

        data_tfs_X = mongo_gen(filter_query = filter_query, key = feature_key, limit = 5)

        pipeline = src.preprocessing.SimplePipeline(
            step_kwargs = [{'batch_size': 1, 'n_threads': 1, 'testing': True},
                           {'collection':[], 'collect_func': None, 'exclude': True},
                           {'use_tfidfs': False}],
            write_stats = False)

        self.assertIsNotNone(pipeline.fit_transform(data_fit_X, data_fit_y))
        self.assertIsNotNone(pipeline.transform(data_tfs_X))
        for word in pipeline.feature_vocabulary_:
            self.assertIn(type(word), [unicode_, unicode])

    @unittest.skip("priotize NBE")
    def test_run_nb(self):
        "Test to see if naive bayes can be used during model_fitting."
        mongo_gen = src.ratings_importer.MongoGenerator
        filter_query = {'style' : 'English Stout'}
        feature_key = 'text'
        target_key = 'taste'

        data_X = mongo_gen(filter_query = None, key = feature_key, limit = 240)
        data_y = mongo_gen(filter_query = None, key = target_key, limit = 240)

        pipeline = src.preprocessing.SimplePipeline(
            step_kwargs= [{'batch_size': 30, 'n_threads': 4, 'testing':True},
                           {'collection':[], 'collect_func': None, 'exclude': True},
                           {'use_tfidfs': False}],
            write_stats = True)

        pipeline_model = src.model_fitting.generic(pipeline = pipeline,
                                                   X = data_X,
                                                   y = data_y,
                                                   model = src.modeling.NBImportances)
        pipeline_model._run()

        # Assert training score is not zero and tokens are meaningful words.
        train_score = pipeline_model.score()
        self.assertTrue(train_score != None and train_score != 0)
        top_10 = pipeline_model.top_tokens()[:10].T[0]
        for token in top_10:
            self.assertIn(type(token), [unicode_, unicode])
