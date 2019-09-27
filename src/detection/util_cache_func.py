#! /usr/bin/env python
# -*- coding: utf8 -*-
import os
import pickle

from time import time

import logging
import hashlib
import gzip

def cache_it(compress=True, skip=None):
    '''
    this is to cache the result of target function 'func' for its first call wtih parameters identified by kwargs.

    cache file name format: func_name-sha1(*args, **kwargs).pickle

    The ret value is same as 'func'.
    '''

    def deco(func):

        def wrapper(*args, **kwargs):

            time_st = time()

            # the folder name for cache result
            CACHE_FOLDER = os.environ.get("XC_CACHE_FUNC_FOLDER", ".cached")
            # mode, working folder (0, None) or folder for current script (1).
            CACHE_FOLDER_MODE = os.environ.get("XC_CACHE_FUNC_PARENT", None)

            # using the working directory folder to cache data.
            if CACHE_FOLDER_MODE is None or CACHE_FOLDER_MODE == 0:
                cache_parent = "."
            elif CACHE_FOLDER_MODE == 1:
                cache_parent = os.path.dirname(os.path.realpath(__file__))
            else:   # suppose it is a folder.
                cache_parent = CACHE_FOLDER_MODE
            cache_folder = os.sep.join([cache_parent, CACHE_FOLDER])

            prefix = func.__name__
            pickle_file = os.sep.join(
                [cache_folder, "{}-{}.pickle".format(prefix, _get_hash(*args, **kwargs))])

            if compress:
                pickle_file += ".gz"

            ret_val = None
            try:
                # folder might not exist.
                if not os.path.exists(cache_folder):
                    os.mkdir(cache_folder)

                if compress:
                    with gzip.open(pickle_file, "rb") as fh:
                        ret_val = pickle.load(fh)
                else:
                    with open(pickle_file, "rb") as fh:
                        ret_val = pickle.load(fh)

                logging.debug("{} is loaded, time used {:.4f}.".format(
                    pickle_file, time()-time_st))

            except:
                ret_val = func(*args, **kwargs)

                if compress:
                    with gzip.open(pickle_file, "wb") as fh:
                        pickle.dump(ret_val, fh)
                else:
                    with open(pickle_file, "wb") as fh:
                        pickle.dump(ret_val, fh)

                logging.debug("{} is generated, time used {:.4f}".format(
                    pickle_file, time() - time_st))

            return ret_val
        return wrapper

    return deco


def _get_hash(*args, **kwargs):
    '''
    concanate all function parameters as a single string and return its hash1 digest.
    '''
    single_string = ""

    for ele in args:
        single_string += "{}".format(ele)

    for key in kwargs:
        single_string += "{}{}".format(key, kwargs[key])

    return hashlib.sha1(single_string.encode("utf-8")).hexdigest()


if __name__ == "__main__":

    @cache_it()
    def abc():
        print("inside function abc")
        return "AAAAA"

    print(abc())
