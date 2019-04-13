#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed March 23 11:27:37 2019

@author: Zeeshan Ali Sayyed
"""

import argparse
import pickle
import os
from scripts.dataset import SegmenterDataset, StandardizerDataset
from scripts.models import CatboostModel, LightGBMModel
from scripts.extractor import StandardizerLE, SegmenterLE, OldFeatureExtractor
from scripts.pipeline import Segmenter, Standardizer

MODELS = "models/"
DEFAULT_SEGMENTER = 'test_segmenter.mod'
DEFAULT_STANDARDIZER = 'test_standardizer.mod'

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
        model = CatboostModel(model_type, options.algorithm_config)
    elif options.algorithm == 'lightgbm':
        model = LightGBMModel(model_type, options.algorithm_config)
    else:
        print("{} is not supported".format(options.algorithm))

    print("Training Model")
    model.train(dataset)
    pred = model.predict(dataset.Xdev)
    print(pred)
    pickle.dump(model, open(os.path.join(MODELS, options.name + '.mod'), 'wb'))


def segment(options):
    print("Called Segment")
    model = pickle.load(open(os.path.join(MODELS, options.model), 'rb'))
    le = SegmenterLE(options.style)
    fe = OldFeatureExtractor(model.feature_template)
    if options.standardize:
        if options.std_model:
            std_model = pickle.load(open(os.path.join(MODELS, options.std_model), 'rb'))
        else:
            std_model = pickle.load(open(os.path.join(MODELS, DEFAULT_STANDARDIZER), 'rb'))
    else:
        std_model = None
    standardizer = Standardizer(std_model, fe, StandardizerLE())
    segmenter = Segmenter(model, fe, le, standardizer)

    if options.mode == 'batch':
        if not options.output:
            options.output = options.input + '.seg'
        segmenter.act_on_file(options.input, options.output)
    elif options.mode == 'interactive':
        print("Type 'exit' to quit the program")
        while True:
            line = input()
            if line == 'exit':
                break
            else:
                print(segmenter.act_on_line(line))
    else:
        print("{} Model is not yet supported".format(options.mode))


def standardize(options):
    print("Called standardize")
    model = pickle.load(open(os.path.join(MODELS, options.model), 'rb'))
    le = StandardizerLE(options.style)
    fe = OldFeatureExtractor(model.feature_template)
    standardizer = Standardizer(model, fe, le)
    if options.mode == 'batch':
        if not options.output:
            options.output = options.input.split('.')[:-1] + '.std'
        standardizer.act_on_file(options.input, options.output)
    elif options.mode == 'interactive':
        print("Type 'exit' to quit the program")
        while True:
            line = input()
            if line == 'exit':
                break
            else:
                print(standardizer.act_on_line(line))
    else:
        print("{} Model is not yet supported".format(options.mode))


def evaluate(options):
    model = pickle.load(open(os.path.join(MODELS, options.model), 'rb'))

def main():
    parser = argparse.ArgumentParser(description="Arabic Segmenter and Orthography Standardizer")
    # parser.add_argument('action', choices=['train', 'segment', 'standardize', 'evaluate'], help="Action to be performed")
    subparsers = parser.add_subparsers()

    # Subparser for train command
    train_parser = subparsers.add_parser('train', help="Train your custom model for a segmenter or a standardizer")
    train_parser.add_argument('type', choices=['segmenter', 'standardizer'], help="What do you want to train? (segmenter/standardizer)")
    train_parser.add_argument('-a', '--algorithm', required=True, choices=['catboost', 'lightgbm'], help="Name of the algorithm")
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
    segment_parser.add_argument('-l', '--model', default='default_segmenter.mod', help="Segmenter model to use")
    segment_parser.add_argument('--style', default='binary_plus', choices=['binary_plus'], help="Segmentation style")
    segment_parser.add_argument('--std-model', default=None, help="Standardizer model to use")
    segment_parser.set_defaults(func=segment)

    # Subparser for standardize command
    std_parser = subparsers.add_parser('standardize', help="Run the standardizer")
    std_parser.add_argument('-i', '--input', help="Input file")
    std_parser.add_argument('-o', '--output', help="Output file. If absent, .std extension std used")
    std_parser.add_argument('-m', '--mode', choices=['interactive', 'batch', 'web'], default='batch')
    std_parser.add_argument('-l', '--model', default='default_standardizer.mod', help="Standardizer model to use")
    std_parser.add_argument('--style', default='eight_class', choices=['eight_class'], help="Standardization style")
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