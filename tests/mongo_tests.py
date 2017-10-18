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

