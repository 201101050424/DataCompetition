# encoding=utf-8
import sys
import argparse
import re
import os
import datetime
import logging
import logging.handlers
import redis
import traceback
import operator
import requests
import bisect
import json
import hashlib
import random
import inspect
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from operator import itemgetter
from tqdm import *
from subprocess import Popen
from subprocess import PIPE
from threading import Lock
from threading import Thread
from urllib import urlencode
from Queue import Queue
from conf import *

logger = logging.getLogger('logger')
if logger.handlers == []:
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.NOTSET)
    logger.addHandler(handler)

    log_file = os.path.basename(__file__).split('.')[0] + '_log'
    handler = logging.FileHandler(filename=log_file, mode='w')
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()

    data = pd.read_csv(data_origin, sep='\t')
    num_total_sample = data.shape[0]
    num_train_sample = int(num_total_sample * split_per)

    data['name'].fillna(value='', inplace=True)
    data['category_name'].fillna(value='', inplace=True)
    data['brand_name'].fillna(value='', inplace=True)
    data['item_description'].fillna(value='', inplace=True)

    all_text = np.hstack(
        [
            data.name.str.lower(),
            data.category_name.str.lower(),
            data.item_description.str.lower(),
        ]
    )
    tokenizer = Tokenizer(100000)
    tokenizer.fit_on_texts(all_text)
    
    for item in ['name', 'category_name', 'item_description']:
        array = tokenizer.texts_to_sequences(data[item].str.lower())
        array = pad_sequences(array, maxlen=20, padding='post')
        np.save('train_data/%s' % item, array)

    for item in ['item_condition_id', 'shipping']:
        logger.debug(item)
        array = data[item].values
        le = LabelEncoder()
        le.fit(array)
        array = le.transform(array)
        array = np.expand_dims(array, axis=1)
        enc = OneHotEncoder()
        enc.fit(array)
        array = enc.transform(array).toarray()
        np.save('train_data/%s' % item, array)
        logger.debug(array.shape)

    logger.debug('brand_name')
    brand_name = data['brand_name'].values
    le = LabelEncoder()
    le.fit(brand_name)
    brand_name = le.transform(brand_name)
    brand_name = np.expand_dims(brand_name, axis=1)
    np.save('train_data/brand_name', brand_name)
    logger.debug(brand_name.shape)

    price = data['price'].values
    price = np.expand_dims(price, axis=1)
    np.save('train_data/%s' % 'price', price)
    logger.debug('price')
    logger.debug(array.shape)
