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

X, y = src.mongo_interface.mongo_loader(limit = 80)
NBEcp = src.modeling.NBExceptional
NBI = src.modeling.NBImportances
generic = src.pipeline_modeling.generic

class TestPipeline(unittest.TestCase):
    def pipeline_test_wrapper(self, model):
        "Complete test of NBE preprocessing. "
        pipeline_model = generic(excludes = [],
                                 min_df = None, max_df = None,
                                 model = model)

        self.assertTrue(type(pipeline_model.cv_score(X, y)) == float)
        self.assertIsNotNone(pipeline_model.get_params())
        pipeline_model.fit(X, y)
        top_10 = pipeline_model.top_tokens()[:10].T[0]
        for token in top_10:
            self.assertIn(type(token), [unicode_, unicode])

    def test_NBEcp_pipeline(self):
        self.pipeline_test_wrapper(NBEcp)

    def test_NBI_pipeline(self):
        self.pipeline_test_wrapper(NBI)

    @unittest.skip("Unimplemented")
    def test_NBE_pipeline(self):
        self.pipeline_test_wrapper(NBE)

    @unittest.skip("Unimplemented")
    def test_NBB_pipeline(self):
        self.pipeline_test_wrapper(NBB)
