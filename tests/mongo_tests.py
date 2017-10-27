'''
Unit tests for everything related to mongo in src/ratings_importer.py
To run: make test_mongo
'''

import unittest
import src

src.load_mongo_interface()

class TestMongoLoader(unittest.TestCase):
    def test_exists_db(self):
        """ Double check to see if the named database and collection exist. """
        mongo_wrapper = src.mongo_interface.MongoGenerator()
        mongo_names = src.mongo_interface.MongoNames()

        self.assertTrue(mongo_names.database in mongo_wrapper.client.database_names())
        self.assertTrue(mongo_names.collection in mongo_wrapper.database.collection_names())

    def test_mongo_loader(self):
        """ Test if the new mongo_loader function works as expeced. """
        mongo_loader = src.mongo_interface.mongo_loader
        X, y = mongo_loader(key = 'style', filter = 'Kvass')
        self.assertEqual(len(X), 297)
        self.assertTrue(type(X.tolist()[0]) == unicode)

    def test_mongo_skip(self):
        """ Test if the mongo interface skips data points with empty text or tastes. """
        mongo_loader = src.mongo_interface.mongo_loader
        X, y = mongo_loader(key = 'style', filter = 'Hefeweizen')
        self.assertTrue(len(X) < 27909)

    def test_expand_query(self):
        """ Test to see if query dictionaries are being correctly expanded into
        SON filter objects. """
        filter_s = {'key_alpha' : ['alpha_a', 'alpha_b'],
                    'key_numeric' : 'num_1'}
        filter_t = {'key_alpha' : ['alpha_a', 'alpha_b'],
                    'key_numeric' : ['num_1']}

        queries = [{'key_alpha' : 'alpha_a', 'key_numeric': 'num_1'},
                   {'key_alpha' : 'alpha_b', 'key_numeric': 'num_1'}]

        queries_s = src.mongo_interface.expand_queries(filter_s)
        for q_e, q_a in zip(sorted(queries_s), sorted(queries)):
            self.assertEqual(q_e, q_a)

        queries_t = src.mongo_interface.expand_queries(filter_t)
        for q_e, q_a in zip(sorted(queries_s), sorted(queries)):
            self.assertEqual(q_e, q_a)

    def test_iter_db(self):
        """ Test if the wrapper is an iterator. """
        #Assumes test_filter_db works
        filter = {'style' : ['Rauchbier'],
                  'ABV' : [5.00]}
        mongo_wrapper = src.mongo_interface.MongoGenerator(filter_query = filter)

        counts = 0
        for review in mongo_wrapper:
            self.assertIsNotNone(review)
            counts += 1

        expected_counts = 64
        self.assertEqual(counts, expected_counts)

    def test_filter_db(self):
        """ Test if filtering works with the wrapper to query the database. """
        filter = {'style' : ['Rauchbier', 'Hefeweizen'],
                  'ABV' : [5.00]}
        mongo_wrapper = src.mongo_interface.MongoGenerator(filter_query = filter)

        expected_counts = 4512 + 64
        self.assertEqual(mongo_wrapper.count(),expected_counts)

    def test_key_db(self):
        """ Test if filtering and key-ing return values works. """
        filter = {'beer/style' : ['Rauchbier'],
                  'beer/ABV' : [5.00]}
        mongo_wrapper = src.mongo_interface.MongoGenerator(filter_query = filter,
                                                            key = 'beer/ABV')

        for beer_abv in mongo_wrapper:
            self.assertEqual(beer_abv, 5.00)

    def test_keys_db(self):
        """ Test if filtering and key-ing multiple return values works. """
        filter = {'style' : ['Rauchbier'],
                  'ABV' : [5.00]}
        res = sorted([5.00, 'Rauchbier'])
        key = ['ABV', 'style']
        mongo_wrapper = src.mongo_interface.MongoGenerator(filter_query = filter,
                                                            key = key)

        for data_point in mongo_wrapper:
            self.assertEqual(tuple(sorted(data_point)), tuple(res))

    def test_limit_db(self):
        """ Test setting limits on arbitrary queries. """
        limit = 10
        mongo_wrapper = src.mongo_interface.MongoGenerator(limit = limit)

        self.assertEqual(len(mongo_wrapper), limit)
        self.assertEqual(len([i for i in mongo_wrapper]), limit)
