"""
ratings_importer holds functions and classes responsible for loading in the ratings data.
"""

from pymongo import MongoClient
from pymongo.uri_parser import parse_uri

class MongoNames():
    """ Holds the names of the mongo database and collection. """
    def __init__(self):
        self.database = u'beer_dump'
        self.collection = u'reviews'

class MongoGenerator(object):
    """ mongo_generator is a wrapper that ultimately encapsulates a pymongo
    curser, modifying its behavior to work with this project. Instantiated
    with filters needed for the query.
    PARAMETERS:
    -----------
        filter : dict, key(string),
                       value(list), (optional; defaults an empty dictionary)
            A dictionary where keys are the names of the keys, and values
            is a list of the possible values they can take. """
    def __init__(self, filter = None):
        self.client = MongoClient()
        self.mongo_names = MongoNames()
        
        self.database = self.client[self.mongo_names.database]
        self.collection = self.database[self.mongo_names.collection]

    def __iter__(self):
        pass

    def count(self):
        pass
