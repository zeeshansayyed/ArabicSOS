#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 25 6:27pm 2019

@author: Zeeshan Ali Sayyed
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder

from scripts.config import catboost_config, lightgbm_config

#For Main Method
from scripts.dataset import SegmenterDataset, StandardizerDataset
import pickle


class Model(ABC):
    """
    The model constructor takes in the following arguments:
    Name: The name of the model file.
    Type: Specifies whether the model is binary or multiclass
    Config: The config of the model. This can be added/edited in config.py
    """

    def __init__(self, type, config='default'):
        self.type = type #Binary or Multiclass
        self.feature_template = None
        self.model = None
        self.style = None

    def __str__(self):
        return "Type: {}; FeatureTemplate: {}".format(self.type, self.feature_template)

    @abstractmethod
    def train(self, dataset):
        self.feature_template = dataset.template
        self.style = dataset.label_style

    @abstractmethod
    def predict(self, X):
        if not self.model:
            raise Exception("Model hasn't been trained yet")

        

class CatboostModel(Model):
    
    def __init__(self, type, config):
        super().__init__(type, config)
        self.config = catboost_config[config]

        
    def train(self, dataset):
        """
        TODO: The train method doesn't support categorical features yet. It assumes 
        that all features are categorical.
        """
        super().train(dataset)
        if self.type == 'binary':
            self.config['loss_function'] = 'Logloss'
            self.config['custom_metric'] = ['Logloss', 'Accuracy', 'F1']
        else:
            self.config['loss_function'] = 'MultiClass'
            self.config['custom_metric'] = ['MultiClass', 'Accuracy', 'TotalF1']
        self.model = CatBoostClassifier(**self.config)
        cat_indices = np.arange(dataset.Xtrain.shape[1])
        eval_set = (dataset.Xdev, dataset.ydev)
        self.model.fit(dataset.Xtrain, dataset.ytrain, cat_features=cat_indices, eval_set=eval_set)

    def predict(self, X):
        super().predict(X)
        return self.model.predict(X)




class LightGBMModel(Model):
    
    def __init__(self, type, config):
        super().__init__(type, config)
        self.config = lightgbm_config[config]
        self.cat_indices = None
        self.encoder = OrdinalEncoder()

    def train(self, dataset):
        super().train(dataset)
        self.cat_indices = list(range(dataset.Xtrain.shape[1]))

        print("Fitting Encoder ...")
        if dataset.Xdev is None:
            self.encoder.fit(np.vstack((dataset.Xtrain, np.array(['UNK'] * dataset.Xtrain.shape[1]))))
        else:
            self.encoder.fit(np.vstack((dataset.Xtrain, dataset.Xdev, np.array(['UNK'] * dataset.Xtrain.shape[1]))))

        print("Encoding Xtrain")
        dataset.Xtrain = self.encoder.transform(dataset.Xtrain)

        print("Encoding Xdev")
        if dataset.Xdev is not None:
            dataset.Xdev = self.encoder.transform(dataset.Xdev)
            print(dataset.Xtrain.shape, dataset.Xdev.shape, dataset.ytrain.shape, dataset.ydev.shape)
        else:
            print(dataset.Xtrain.shape, dataset.ytrain.shape)

        if self.type == 'binary':
            self.config['objective'] = 'binary'
            self.config['metric'] = 'binary_logloss'
        else:
            self.config['objective'] = 'multiclass'
            self.config['metric'] = 'multi_logloss'
            self.config['num_class'] = len(np.unique(dataset.ytrain))

        print("Creating lgb dataset")
        lgb_train = lgb.Dataset(dataset.Xtrain, dataset.ytrain)
        lgb_eval = lgb.Dataset(dataset.Xdev, dataset.ydev, reference=lgb_train)
        print("Training LGB")
        self.model = lgb.train(self.config, lgb_train, valid_sets=[lgb_train, lgb_eval])

    def predict(self, X):
        super().predict(X)
        if type(X) == list:
            if type(X[0]) == list:
                X = np.array(X)
            else:
                X = np.array(X).reshape(-1, len(self.feature_template))

        if X.dtype != 'float64':
            for col_no in range(len(self.encoder.categories_)):
                X[np.isin(X[:, col_no], self.encoder.categories_[col_no], invert=True), col_no] = 'UNK'
            X = self.encoder.transform(X)
        else:
            raise Exception("dtype of input array is {}. This should not be!".format(X.dtype))

        if self.type == 'binary':
            return self.model.predict(X) > 0.5
        else:
            return np.argmax(self.model.predict(X), axis=1)


if __name__ == "__main__":
    # sd = SegmenterDataset('t1', 'sample/small_sample.seg', None, name='test')
    # print(sd.type, sd.name, sd.template)
    # print(sd)
    # d1 = SegmenterDataset('t1', 'data/segmenter/train4.txt', 'data/segmenter/dev1.txt')
    # d1 = pickle.load(open('data/standardizer/std.ds', 'rb'))
    # model = LightGBMModel('multiclass', 'default')
    # model.train(d1)
    # print(model)
    # pickle.dump(model, open('models/std_test.mod', 'wb'))
    model = pickle.load(open('models/std_test.mod', 'rb'))
    d2 = pickle.load(open('data/standardizer/siyar.ds', 'rb'))
    print(model.predict(d2.Xdev))
