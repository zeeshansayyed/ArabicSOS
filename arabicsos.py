#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed March 23 11:27:37 2019

@author: Zeeshan Ali Sayyed
"""

import argparse
import pickle
import os
from scripts.trainer import SegmenterDataset, StandardizerDataset, CatboostModel, LightGBMModel

MODELS = "models/"

def train(options):
    if options.type == 'segmenter':
        dataset = SegmenterDataset(options.template, options.train, options.dev, options.test, options.name)
        model_type = 'binary'
    elif options.type == 'standardizer':
        dataset = StandardizerDataset(options.template, options.train, options.dev, options.test, options.name)
        model_type = 'multiclass'
    else:
        print("{} is not supported".format(options.type))

    if options.algorithm == 'catboost':
        model = CatboostModel(model_type, options.template, options.algorithm_config)
    elif options.algorithm == 'lightgbm':
        model = LightGBMModel(model_type, options.template, options.algorithm_config)
    else:
        print("{} is not supported".format(options.algorithm))

    print("Training Model")
    model.train(dataset)
    pickle.dump(model, open(os.path.join(MODELS, options.name + '.mod'), 'wb'))


def segment(options):
    print("Called Segment")
    model = pickle.load(open(os.path.join(MODELS, options.model), 'rb'))

def standardize(options):
    print("Called standardize")
    print(options)

def evaluate(options):
    print("Called Evaluate")
    print(options)

def main():
    parser = argparse.ArgumentParser(description="Arabic Segmenter and Orthography Standardizer")
    # parser.add_argument('action', choices=['train', 'segment', 'standardize', 'evaluate'], help="Action to be performed")
    subparsers = parser.add_subparsers()

    # Subparser for train command
    train_parser = subparsers.add_parser('train', help="Train your custom model for a segmenter or a standardizer")
    train_parser.add_argument('type', choices=['segmenter', 'standardizer'], help="What do you want to train? (segmenter/standardizer)")
    train_parser.add_argument('-a', '--algorithm', choices=['catboost', 'lightgbm'], help="Name of the algorithm")
    train_parser.add_argument('-c', '--algorithm-config', default='default', help="Configuration for the training algorithm from config.py")
    train_parser.add_argument('-t', '--train', required=True, help='train location')
    train_parser.add_argument('-d', '--dev', required=True, help='dev location')
    train_parser.add_argument('-s', '--test', default=None, help='test location')
    train_parser.add_argument('-n', '--name', help='Name of the resultant model. It will be saved in /models.')
    train_parser.add_argument('-p', '--template', default='t1', help="Feature template to use")
    train_parser.set_defaults(func=train)

    # Subparser for segment command
    segment_parser = subparsers.add_parser('segment', help="Run the segmenter")
    # segment_parser.add_argument('--standardize', help="Pass the model that will be used to standardize the text before segmenting. If no model is passed, the default standardizer model will be used")
    segment_parser.add_argument('-i', '--input', help="Input file")
    segment_parser.add_argument('-o', '--output', help="Output file. If absent, .seg extension will be used")
    segment_parser.add_argument('-s', '--standardize', action='store_true', help="Standardize the file before segmenting")
    segment_parser.add_argument('-m', '--mode', choices=['interactive', 'batch', 'web'], default='batch')
    segment_parser.add_argument('-l', '-model', type=argparse.FileType('r'), help="Segmenter model to use")
    segment_parser.add_argument('--std-model', type=argparse.FileType('r'), help="Standardizer model to use")

    segment_parser.set_defaults(func=segment)

    # Subparser for standardize command
    std_parser = subparsers.add_parser('standardize', help="Run the standardizer")
    std_parser.add_argument('-i', '--input', type=argparse.FileType('r'), help="Input file")
    std_parser.add_argument('-o', '--output', type=argparse.FileType('w'), help="Output file. If absent, .std extension std used")
    std_parser.add_argument('-m', '--mode', choices=['interactive', 'batch', 'web'])
    std_parser.add_argument('-l', '-model', type=argparse.FileType('r'), help="Standardizer model to use")
    std_parser.set_defaults(func=standardize)

    # Subparser for evaluate command
    eval_parser = subparsers.add_parser('evaluate', help="Evaluate the learned segmenter/standardizer models on the given test files")
    eval_parser.add_argument('what', choices=['segmenter', 'standardizer'], help="What do you want to evaluate? (Segmenter/Standardizer)")
    eval_parser.add_argument('-l', '--model', type=argparse.FileType('r'), help="Path to model file")
    eval_parser.add_argument('-t', '--test', nargs='+', type=argparse.FileType('r'), help='List of all files to be evaluated')
    eval_parser.set_defaults(func=evaluate)

    options = parser.parse_args()
    options.func(options)

if __name__ == "__main__":
    main()