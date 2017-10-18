'''
Unit tests for the entire pipeline.
Make test_pipeline to run
'''

from numpy import array
import unittest
import src

src.load_preprocessing()
src.load_ratings_importer()

class TestPipeline(unittest.TestCase):
    def test_loaded(self):
        "All required modules are loaded"
        self.assertIsNotNone(src.ratings_importer.MongoGenerator)
        self.assertIsNotNone(src.preprocessing.SimplePipeline)

    @unittest.skip("pipeline not implemented")
    def test_preprocessing_pipe(self):
        "Mongo to vector matrix test"

        filter_query = {'beer/style' : 'Rauchbier'}
        keys = ["review/taste", "review/text"]
        data_f = src.ratings_importer.MongoGenerator(filter_query = None,
                                                     key = keys,
                                                     limit = 10)
        data_f = array(list(data_f))
        data_t = src.ratings_importer.MongoGenerator(filter_query = filter_query,
                                                     key = "review/text",
                                                     limit = 10)

        pipeline = src.preprocessing.SimplePipeline(
            steps = [src.preprocessing.DocTokenizer(batch_size = 1,
                                                    n_threads = 1, 
                                                    testing = True),
                     src.preprocessing.TokenFilter(collection = [],
                                                   collect_func = None,
                                                   exclude = True),
                     src.preprocessing.TemTokenPreprocessor(threshold = 0.5,
                                                            metric = None),
                     src.preprocessing.TokenVectorizer(use_tfidfs = False)])

        self.assertIsNotNone(pipeline.fit_transform(data_f[:,0], data_f.T[1]))
        self.assertIsNotNone(pipeline.transform(data_t))
