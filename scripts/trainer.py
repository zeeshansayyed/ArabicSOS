#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 25 6:27pm 2019

@author: Zeeshan Ali Sayyed
"""

from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from scripts.extractor import OldFeatureExtractor, SegmenterLE, StandardizerLE
from scripts.preprocessor import clean_arabic
from scripts.util import get_num_lines
from scripts.config import catboost_config, lightgbm_config
import pudb;
import pandas as pd


class Dataset(ABC):

    def __init__(self, type, template, label_style, train_file, dev_file, test_file, name):
        self.type = type
        self.name = name
        self.template = template
        self.fe = OldFeatureExtractor(self.template)
        self.label_style = label_style
        if self.type == 'segmenter':
            self.le = SegmenterLE(label_style)
        elif self.type == 'standardizer':
            self.le = StandardizerLE(label_style)
        if train_file:
            print("Creating train set...")
            self.Xtrain, self.ytrain = self.create_dataset(train_file)
        if dev_file:
            print("Creating dev set...")
            self.Xdev, self.ydev = self.create_dataset(dev_file)
        if test_file:
            print("Creating test set...")
            self.Xtest, self.ytest = self.create_dataset(test_file)

    def __str__(self):
        return "Type: {}, Name: {}, Template: {}, Label Style: {}".format(self.type, self.name, self.template, self.label_style)

    @abstractmethod
    def extract(self, line):
        pass

    def create_dataset(self, filename):
        X, y = [], []
        with open(filename, 'r') as infile:
            for line in tqdm(infile, total=get_num_lines(filename)):
                features, labels = self.extract(line)
                X += features
                y += labels
        return np.array(X), np.array(y)


class SegmenterDataset(Dataset):
    """
    The constructor takes the segmented file as input.
    It will automatically extract the features for characters
    according to the specified template and labels for the corresponding
    characters according to the specified style and create number arrays for X and y
    """

    def __init__(self, template, train_file, dev_file, test_file=None, name=None, label_style='binary_plus'):
        super().__init__('segmenter', template, label_style, train_file, dev_file, test_file, name)
        self.le = SegmenterLE()

    def extract(self, line):
        segline = clean_arabic(line, contains_plus=True).strip()
        rawline = segline.replace('+', '')
        sent_features = self.fe.sent_to_features(rawline.split(), flat=True)
        sent_labels = self.le.sent_to_labels(segline.split())
        return sent_features, sent_labels


class StandardizerDataset(Dataset):
    """SSO stands for Sub Standard Orthography"""

    def __init__(self, template, train_file, dev_file, test_file=None, name=None, label_style='eight_class'):
        self.substandard_dict = {
            'أ': 'ا',
            'إ': 'ا',
            'آ': 'ا',
            'ة': 'ه',
        }
        self.substandard_chars = set(('ا', 'ه'))
        super().__init__('standardizer', template, label_style, train_file, dev_file, test_file, name)

    def substandardize(self, word):
        word = list(word)
        for pos, char in enumerate(word):
            if char in self.substandard_dict:
                word[pos] = self.substandard_dict[char]
        if word[-1] == 'ي':
            word[-1] = 'ى'

        return ''.join(word)

    def extract(self, line):
        stdline = clean_arabic(line).strip().split()
        ssoline = [self.substandardize(w) for w in stdline]
        features, labels = [], []
        for word_pos, word in enumerate(ssoline):
            for char_pos, char in enumerate(word):
                if char in self.substandard_chars or (char_pos == len(word)-1 and char == 'ى'):
                    features.append(self.fe.char_to_features(char_pos, word_pos, ssoline))
                    labels.append(self.le.char_to_label(stdline[word_pos][char_pos]))
        return features, labels




class Model(ABC):
    """
    The model constructor takes in the following arguments:
    Name: The name of the model file.
    Type: Specifies whether the model is binary or multiclass
    Config: The config of the model. This can be added/edited in config.py
    Template: The feature template used to create the data.
    """

    def __init__(self, type, template='t1', config='default'):
        self.type = type #Binary or Multiclass
        self.feature_template = template
        self.model = None

    def __str__(self):
        return "Type: {}; FeatureTemplate: {}".format(self.type, self.feature_template)

    @abstractmethod
    def train(self, dataset):
        pass

    @abstractmethod
    def predict(self, X):
        if not self.model:
            raise Exception("Model hasn't been trained yet")

class CatboostModel(Model):
    
    def __init__(self, type, template, config):
        super().__init__(type, config, template)
        self.config = catboost_config[config]

        
    def train(self, dataset):
        """
        TODO: The train method doesn't support categorical features yet. It assumes 
        that all features are categorical.
        """
        if self.type == 'binary':
            self.config['loss_function'] = 'Logloss'
            self.config['custom_metric'] = ['Logloss', 'Accuracy', 'F1']
        else:
            self.config['loss_function'] = 'MultiClass'
            self.config['custom_metric'] = ['Multiclass', 'Accuracy', 'TotalF1']
        self.model = CatBoostClassifier(**self.config)
        cat_indices = np.arange(dataset.Xtrain.shape[1])
        eval_set = (dataset.Xdev, dataset.ydev)
        self.model.fit(dataset.Xtrain, dataset.ytrain, cat_features=cat_indices, eval_set=eval_set)

    def predict(self, X):
        super().predict(X)
        return self.model.predict(X)




class LightGBMModel(Model):
    
    def __init__(self, type, template, config):
        super().__init__(type, template)
        self.config = lightgbm_config[config]
        self.cat_indices = None
        self.encoders = None

    def train(self, dataset):
        self.cat_indices = list(range(dataset.Xtrain.shape[1]))
        dataset.Xtrain = pd.DataFrame(dataset.Xtrain)
        dataset.Xdev = pd.DataFrame(dataset.Xdev)
        for X in [dataset.Xtrain, dataset.Xdev]:
            for column in self.cat_indices:
                X[column] = X[column].astype('category')

        print(dataset.Xtrain.shape, dataset.Xdev.shape, dataset.ytrain.shape, dataset.ydev.shape)

        if self.type == 'binary':
            self.config['objective'] = 'binary'
            self.config['metric'] = 'binary_logloss'
        else:
            self.config['objective'] = 'multiclass'
            self.config['metric'] = 'multi_logloss'
            self.config['num_class'] = len(np.unique(dataset.ytrain))
        
        lgb_train = lgb.Dataset(dataset.Xtrain, dataset.ytrain)
        lgb_eval = lgb.Dataset(dataset.Xdev, dataset.ydev, reference=lgb_train)
        # self.model = LGBMClassifier(**self.config)
        # eval_set = (dataset.Xdev, dataset.ydev)
        # self.model.fit(dataset.Xtrain, dataset.ytrain, eval_set)
        self.model = lgb.train(self.config, lgb_train, valid_sets=[lgb_train, lgb_eval])

    def predict(self, X):
        super().predict(X)
        X = pd.DataFrame(X)
        for column in self.cat_indices:
            X[column] = X[column].astype('category')
        if self.type == 'binary':
            return self.model.predict(X) > 0.5
        else:
            return np.argmax(self.model.predict(X), axis=1)


if __name__ == "__main__":
    sd = SegmenterDataset('t1', 'sample/small_sample.seg', None, name='test')
    print(sd.type, sd.name, sd.template)
    print(sd)
