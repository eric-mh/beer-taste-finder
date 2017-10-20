'''
Unit tests for the entire pipeline.
Also tests individual model_fitting approaches in src/model_fitting.py
Make test_pipeline to run
'''
from numpy import array, unicode_
import unittest
import src

src.load_preprocessing()
src.load_ratings_importer()
src.load_model_fitting()


class TestPipeline(unittest.TestCase):
    def test_loaded(self):
        "All required modules are loaded"
        self.assertIsNotNone(src.ratings_importer.MongoGenerator)
        self.assertIsNotNone(src.preprocessing.SimplePipeline)
        self.assertIsNotNone(src.modeling.LinearImportances)

    def test_preprocessing_pipe(self):
        "Mongo to vector matrix test"
        mongo_gen = src.ratings_importer.MongoGenerator

        filter_query = {'beer/style' : 'Rauchbier'}
        feature_key = 'review/text'
        target_key = 'review/taste'

        data_fit_X = mongo_gen(filter_query = None, key = feature_key, limit = 10)
        data_fit_y = mongo_gen(filter_query = None, key = target_key, limit = 10)

        data_tfs_X = mongo_gen(filter_query = filter_query, key = feature_key, limit = 5)

        pipeline = src.preprocessing.SimplePipeline(
            step_kwargs = [{'batch_size': 1, 'n_threads': 1, 'testing': True},
                           {'collection':[], 'collect_func': None, 'exclude': True},
                           {'threshold': 0.5, 'metric': None},
                           {'use_tfidfs': False}],
            write_stats = False)

        self.assertIsNotNone(pipeline.fit_transform(data_fit_X, data_fit_y))
        self.assertIsNotNone(pipeline.transform(data_tfs_X))
        for word in pipeline.feature_vocabulary_:
            self.assertIn(type(word), [unicode_, unicode])

    def test_run_linear(self):
        "Test to see if the basic linear pipeline can give results."
        mongo_gen = src.ratings_importer.MongoGenerator
        filter_query = {'beer/style' : 'English Stout'}
        feature_key = 'review/text'
        target_key = 'review/taste'

        data_X = mongo_gen(filter_query = None, key = feature_key, limit = 3000)
        data_y = mongo_gen(filter_query = None, key = target_key, limit = 3000)

        pipeline = src.preprocessing.SimplePipeline(
            step_kwargs= [{'batch_size': 150, 'n_threads': 4, 'testing':True},
                           {'collection':[], 'collect_func': None, 'exclude': True},
                           {'threshold': 0.5, 'metric': None},
                           {'use_tfidfs': False}],
            write_stats = True)

        pipeline_model = src.model_fitting.linear(pipeline = pipeline,
                                                  X = data_X,
                                                  y = data_y,
                                                  model = src.modeling.LinearImportances)
        pipeline_model._run()

        # Assert training score is not zero and tokens are meaningful words.
        train_score = pipeline_model.score()
        self.assertTrue(train_score != None and train_score != 0)
        top_10 = pipeline_model.top_tokens()[:10].T[0]
        for token in top_10:
            self.assertIn(type(token), [unicode_, unicode])
