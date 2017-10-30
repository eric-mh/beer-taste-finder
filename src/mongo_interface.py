"""
ratings_importer holds functions and classes responsible for loading in the ratings data.
"""

from pymongo import MongoClient
from pymongo.uri_parser import parse_uri
from sklearn.model_selection import ParameterGrid

from numpy import array
from itertools import chain

def expand_queries(filter_query):
    """ Expands a dict filter into the needed SON filters for the pymongo querying.
    Uses sklearn.model_selection's ParameterGrid to enumerate combinations. """
    if filter_query is None:
        return [None]
    for key in filter_query.keys():
        if type(filter_query[key]) != list:
            filter_query[key] = [filter_query[key]]
    return list(ParameterGrid(filter_query))

def mongo_loader(key = None, filter = None, feature = 'text', target = 'taste',
                 limit = None):
    if key == None or filter == None:
        filter_query = None
    else:
        filter_query = {key: filter}
    mongo_gen = MongoGenerator(filter_query = filter_query,
                               key = [feature, target], limit = limit)
    return array(list(mongo_gen)).T

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
    def __init__(self, filter_query = None, key = None, limit = None):
        self.client = MongoClient()
        self.mongo_names = MongoNames()
        self.limit = limit
        
        self.database = self.client[self.mongo_names.database]
        self.collection = self.database[self.mongo_names.collection]
        self.key = key

        self.cursors = []
        for query in expand_queries(filter_query):
            self.cursors.append(self.collection.find(query))

        self.chained_cursors = chain(*self.cursors)

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.count()

    def __iter__(self):
        return self

    def next(self):
        # Checking stop iteration from limit
        if self.limit is not None:
            if self.limit == 0:
                raise StopIteration
            else:
                self.limit -= 1
        # Checking if there's specific keys being queried.
        if self.key is None:
            return self.chained_cursors.next()
        else:
            return self._skey_return(self.chained_cursors.next(),self.key)

    def _skey_return(self,dict, keys):
        "Special key indexer that deals with dictionaries possibly not having an entry"
        if type(keys) != list:
            keys = [keys]
        if array([(key in dict.keys()) for key in keys]).min() == False:
            return self.next() # Skip data point of key is missing
        else:
            return [dict[key] for key in keys]

    def count(self):
        if self.limit is None:
            return sum([cursor.count() for cursor in self.cursors])
        else:
            return self.limit
