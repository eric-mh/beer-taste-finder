'''
Unit tests for everything related to mongo in src/ratings_importer.py
'''

import unittest
import src

class TestMongoLoader(unittest.TestCase):
    def test_exists_db(self):
        """ Double check to see if the named database and collection exist. """
        mongo_wrapper = src.ratings_importer.MongoGenerator()
        mongo_names = src.ratings_importer.MongoNames()

        self.assertTrue(mongo_names.database in mongo_wrapper.client.database_names())
        self.assertTrue(mongo_names.collection in mongo_wrapper.database.collection_names())

    def test_expand_query(self):
        filter_s = {'key_alpha' : ['alpha_a', 'alpha_b'],
                    'key_numeric' : 'num_1'}
        filter_t = {'key_alpha' : ['alpha_a', 'alpha_b'],
                    'key_numeric' : ['num_1']}

        queries = [{'key_alpha' : 'alpha_a', 'key_numeric': 'num_1'},
                   {'key_alpha' : 'alpha_b', 'key_numeric': 'num_1'}]

        queries_s = src.ratings_importer.expand_queries(filter_s)
        for q_e, q_a in zip(sorted(queries_s), sorted(queries)):
            self.assertEqual(q_e, q_a)

        queries_t = src.ratings_importer.expand_queries(filter_t)
        for q_e, q_a in zip(sorted(queries_s), sorted(queries)):
            self.assertEqual(q_e, q_a)

    def test_iter_db(self):
        #Assumes test_filter_db works
        filter = {'beer/style' : ['Rauchbier'],
                  'beer/ABV' : ['5.00']}
        mongo_wrapper = src.ratings_importer.MongoGenerator(filter)

        counts = 1
        for review in mongo_wrapper:
            self.assertIsNotNone(review)
            counts += 1

        expected_counts = 64
        self.assertEqual(counts, expected_counts)

    def test_filter_db(self):
        filter = {'beer/style' : ['Rauchbier', 'Hefeweizen'],
                  'beer/ABV' : ['5.00']}
        mongo_wrapper = src.ratings_importer.MongoGenerator(filter)

        expected_counts = 4511 + 64
        self.assertEqual(mongo_wrapper.count(),expected_counts)
