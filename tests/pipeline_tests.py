'''
Unit tests for the entire pipeline.
Also tests individual model_fitting approaches in src/model_fitting.py
Make test_pipeline to run
'''
from numpy import array, unicode_
import unittest
import src

src.load_modeling()
src.load_pipeline_model()
src.load_mongo_interface()
src.load_preprocessing()

X, y = src.mongo_interface.mongo_loader(limit = 80)
NBI = src.modeling.NBImportances
NBX = src.modeling.NBExceptional
generic = src.pipeline_modeling.generic

cv_scorer = src.pipeline_modeling.cv_scorer

class TestPipeline(unittest.TestCase):
    def pipeline_test_wrapper(self, model):
        "Complete test of NBE preprocessing. "
        pipeline_model = generic(excludes = [],
                                 min_df = None, max_df = None,
                                 model = model)

        self.assertIsNotNone(pipeline_model.get_params())
        pipeline_model.fit(X, y)
        self.assertTrue(pipeline_model.score(X, y) != 0)
        top_10 = pipeline_model.top_tokens()[:10].T[0]
        for token in top_10:
            self.assertIn(type(token), [unicode_, unicode])
        self.assertTrue(cv_scorer(pipeline_model, X, y).mean() != 0)
    
    def test_early_preprocessor(self):
        pipeline_const = src.preprocessing.SimplePipeline
        pipeline = pipeline_const([{'batch_size' : 1, 'n_threads' : 1, 'testing' : True},
                                   {'collection' : []},
                                   {'use_tfidfs' : False,
                                    'max_df' : 1.0, 'min_df' : 1}],
                                  write_stats = False)
        self.assertIsNotNone(pipeline.fit_transform(X[:10]))
        self.assertIsNotNone(pipeline.transform(X))

    @unittest.skip("Only included for completeness")
    def test_NBEcp_pipeline(self):
        self.pipeline_test_wrapper(NBI)

    def test_NBI_pipeline(self):
        self.pipeline_test_wrapper(NBX)

    @unittest.skip("Unimplemented")
    def test_NBE_pipeline(self):
        self.pipeline_test_wrapper(NBE)

    @unittest.skip("Unimplemented")
    def test_NBB_pipeline(self):
        self.pipeline_test_wrapper(NBB)
