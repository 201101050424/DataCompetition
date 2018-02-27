# encoding=utf-8
import sys
import argparse
import numpy as np
import re
import math
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
import numpy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import csv

from operator import itemgetter
from tqdm import *
from subprocess import Popen
from subprocess import PIPE
from threading import Lock
from threading import Thread
from urllib import urlencode
from Queue import Queue
from conf import *
from gensim.models import KeyedVectors
from torch.autograd import Variable

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


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        self.data_1 = torch.from_numpy(
            np.load('val_data_1.npy')).type(torch.LongTensor)
        self.data_2 = torch.from_numpy(
            np.load('val_data_2.npy')).type(torch.LongTensor)
        self.labels = torch.from_numpy(
            np.load('val_labels.npy')).type(torch.LongTensor)

    def __len__(self):
        return self.data_1.shape[0]

    def __getitem__(self, idx):
        q1 = self.data_1[idx]
        q2 = self.data_2[idx]
        label = self.labels[idx]
        return q1, q2, label


class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self):
        self.name = torch.from_numpy(
            np.load('./train_data/name.npy')).type(torch.LongTensor)
        self.item_condition_id = torch.from_numpy(
            np.load('./train_data/item_condition_id.npy')).type(torch.LongTensor)
        self.category_name = torch.from_numpy(
            np.load('./train_data/category_name.npy')).type(torch.LongTensor)
        self.brand_name = torch.from_numpy(
            np.load('./train_data/brand_name.npy')).type(torch.LongTensor)
        self.price = torch.from_numpy(
            np.load('./train_data/price.npy')).type(torch.LongTensor)
        self.shipping = torch.from_numpy(
            np.load('./train_data/shipping.npy')).type(torch.LongTensor)
        self.item_description = torch.from_numpy(
            np.load('./train_data/item_description.npy')).type(torch.LongTensor)

    def __len__(self):
        return self.name.shape[0]

    def __getitem__(self, idx):
        return self.name[idx], self.item_condition_id[idx], self.category_name[idx], self.brand_name[idx], self.shipping[idx], self.price[idx], self.item_description[idx]


class ModelLSTM(nn.Module):
    def __init__(self):
        super(ModelLSTM, self).__init__()
        num_word = 90000
        num_dim = 20
        num_brand_name = 5000
        self.embedding_name = nn.Embedding(
            num_word, num_dim)
        self.embedding_category_name = nn.Embedding(
            num_word, num_dim)
        self.embedding_item_description = nn.Embedding(
            num_word, num_dim)
        self.embedding_brand_name = nn.Embedding(
            num_brand_name, num_dim)

        hidden_size = 100
        self.lstm_name = nn.LSTM(
            num_dim, hidden_size, batch_first=True)
        self.lstm_category_name = nn.LSTM(
            num_dim, hidden_size, batch_first=True)
        self.lstm_item_description = nn.LSTM(
            num_dim, hidden_size, batch_first=True)

        self.dropout_cat = nn.Dropout(0.5)
        self.bn_cat = nn.BatchNorm1d(326)

        self.fc1 = nn.Linear(326, 326)
        self.bn_fc1 = nn.BatchNorm1d(326)

        self.fc3 = nn.Linear(326, 1)

    def forward(self, name, item_condition_id, category_name, brand_name, shipping, price, item_description):

        name = self.embedding_name(name)
        name, _ = self.lstm_name(name)
        name = name[:, -1, :]

        category_name = self.embedding_category_name(category_name)
        category_name, _ = self.lstm_category_name(category_name)
        category_name = category_name[:, -1, :]

        item_description = self.embedding_item_description(item_description)
        item_description, _ = self.lstm_item_description(item_description)
        item_description = item_description[:, -1, :]

        brand_name = self.embedding_brand_name(brand_name)
        brand_name = torch.squeeze(brand_name, 1)

        x = torch.cat([name, item_condition_id.float(), category_name,
                       brand_name, shipping.float(), item_description], 1)

        x = self.dropout_cat(x)
        x = self.bn_cat(x)
        x = F.relu(x)

        x = self.bn_fc1(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x


if __name__ == "__main__":
    data_loader = torch.utils.data.DataLoader(
        DatasetTrain(), batch_size=batch_size, shuffle=True)

    model = ModelLSTM().cuda()
    logger.debug(model)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for i in range(epoch_num):
        model.train()
        train_loss_list = []
        for batch in tqdm(data_loader):
            name = batch[0].cuda()
            item_condition_id = batch[1].cuda()
            category_name = batch[2].cuda()
            brand_name = batch[3].cuda()
            shipping = batch[4].cuda()
            price = batch[5].float().cuda()
            item_description = batch[6].cuda()

            output = model(
                Variable(name),
                Variable(item_condition_id),
                Variable(category_name),
                Variable(brand_name),
                Variable(shipping),
                Variable(price),
                Variable(item_description),
            )
            # logger.debug(type(output.data))

            price = Variable(torch.squeeze(price, 1))

            optimizer.zero_grad()
            loss = F.mse_loss(output, price)
            loss.backward()
            optimizer.step()
            logger.debug(loss.data[0])
            # train_loss_list.append(loss.data[0])

        # model.eval()
        # val_loss_list = []
        # for batch in (val_data_loader):
        #     input1 = batch[0].cuda()
        #     input2 = batch[1].cuda()
        #     label = batch[2].cuda()

        #     output = model(Variable(input1), Variable(input2))
        #     label = Variable(label)

        #     loss = F.nll_loss(output, label, class_weight)
        #     val_loss_list.append(loss.data[0])

        # avg_train_loss = sum(train_loss_list) * 1.0 / len(train_loss_list)
        # avg_val_loss = sum(val_loss_list) * 1.0 / len(val_loss_list)

        # print 'epoch:%d avg_train_loss=%lf avg_val_loss=%lf' % (i, avg_train_loss, avg_val_loss)
        # if i % model_save_interval == 0:
        #     torch.save(model, 'model_lstm/model_%d' % (i))
