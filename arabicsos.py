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
from scripts.extractor import StandardizerLE, SegmenterLE, OldFeatureExtractor, SegmenterDatasetHelper, \
    StandardizerDatasetHelper
from scripts.pipeline import Segmenter, Standardizer

MODELS = "models/"
DEFAULT_SEGMENTER = 'default_segmenter.mod'
DEFAULT_STANDARDIZER = 'test_standardizer.mod'
DEFAULT_SEGMENTER_STYLE = 'binary_plus'
DEFAULT_STANDARDIZER_STYLE = 'eight_class'


def train(options):
    if options.type == 'segmenter':
        model_type = 'binary'
        if options.dataset:
            dataset = pickle.load(open(options.dataset, 'rb'))
        else:
            if not options.label_style: style = DEFAULT_SEGMENTER_STYLE
            dataset = SegmenterDataset(options.template, options.train, options.dev, options.test, options.name,
                                       label_style=style)
    elif options.type == 'standardizer':
        model_type = 'multiclass'
        if options.dataset:
            dataset = pickle.load(open(options.dataset, 'rb'))
        else:
            if not options.label_style: style = DEFAULT_STANDARDIZER_STYLE
            dataset = StandardizerDataset(options.template, options.train, options.dev, options.test, options.name,
                                          label_style=options.label_style)
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
    # pred = model.predict(dataset.Xdev)
    # print(pred)
    pickle.dump(model, open(os.path.join(MODELS, options.name + '.mod'), 'wb'))


def segment(options):
    print("Called Segment")
    model = pickle.load(open(os.path.join(MODELS, options.model), 'rb'))
    le = SegmenterLE(model.style)
    fe = OldFeatureExtractor(model.feature_template)
    helper = SegmenterDatasetHelper(model.style)
    if options.standardize:
        if options.std_model:
            std_model = pickle.load(open(os.path.join(MODELS, options.std_model), 'rb'))
        else:
            std_model = pickle.load(open(os.path.join(MODELS, DEFAULT_STANDARDIZER), 'rb'))
        standardizer = Standardizer(std_model, fe, StandardizerLE(std_model.style),
                                    StandardizerDatasetHelper(std_model.style))
    else:
        standardizer = None

    segmenter = Segmenter(model, fe, le, helper, standardizer)

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
    le = StandardizerLE(model.style)
    fe = OldFeatureExtractor(model.feature_template)
    helper = StandardizerDatasetHelper(model.style)
    standardizer = Standardizer(model, fe, le, helper)
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
    fe = OldFeatureExtractor(model.feature_template)
    if options.what == 'segmenter':
        le = SegmenterLE(model.style)
        helper = SegmenterDatasetHelper(model.style)
        actor = Segmenter(model, fe, le, helper)
    else:
        le = StandardizerLE(model.style)
        helper = StandardizerDatasetHelper(model.style)
        actor = Standardizer(model, fe, le, helper)
    for file in options.files:
        error_list, result = actor.evaluate_file(file)
        print(result)


def main():
    parser = argparse.ArgumentParser(description="Arabic Segmenter and Orthography Standardizer")
    subparsers = parser.add_subparsers()

    # Subparser for train command
    train_parser = subparsers.add_parser('train', help="Train your custom model for a segmenter or a standardizer")
    train_parser.add_argument('type', choices=['segmenter', 'standardizer'],
                              help="What do you want to train? (segmenter/standardizer)")
    train_parser.add_argument('-a', '--algorithm', required=True, choices=['catboost', 'lightgbm'],
                              help="Name of the algorithm")
    train_parser.add_argument('-c', '--algorithm-config', default='default',
                              help="Configuration for the training algorithm from config.py")
    train_parser.add_argument('-t', '--train', help='train location')
    train_parser.add_argument('-d', '--dev', help='dev location')
    train_parser.add_argument('-s', '--test', default=None, help='test location')
    train_parser.add_argument('--dataset', default=None, help='Dataset pickle path')
    train_parser.add_argument('-n', '--name', help='Name of the resultant model. It will be saved in /models.')
    train_parser.add_argument('-p', '--template', default='t1', help="Feature template to use")
    train_parser.add_argument('--label-style',
                              help="Currently supported: eight_class for standardizer and binary_plus for segmenter")
    train_parser.set_defaults(func=train)

    # Subparser for segment command
    segment_parser = subparsers.add_parser('segment', help="Run the segmenter")
    segment_parser.add_argument('-i', '--input', help="Input file")
    segment_parser.add_argument('-o', '--output', help="Output file. If absent, .seg extension will be used")
    segment_parser.add_argument('-s', '--standardize', action='store_true',
                                help="Standardize the file before segmenting")
    segment_parser.add_argument('-m', '--mode', choices=['interactive', 'batch', 'web'], default='batch')
    segment_parser.add_argument('-l', '--model', default=DEFAULT_SEGMENTER, help="Segmenter model to use")
    segment_parser.add_argument('--std-model', default=None, help="Standardizer model to use")
    segment_parser.set_defaults(func=segment)

    # Subparser for standardize command
    std_parser = subparsers.add_parser('standardize', help="Run the standardizer")
    std_parser.add_argument('-i', '--input', help="Input file")
    std_parser.add_argument('-o', '--output', help="Output file. If absent, .std extension std used")
    std_parser.add_argument('-m', '--mode', choices=['interactive', 'batch', 'web'], default='batch')
    std_parser.add_argument('-l', '--model', default=DEFAULT_STANDARDIZER, help="Standardizer model to use")
    std_parser.set_defaults(func=standardize)

    # Subparser for evaluate command
    eval_parser = subparsers.add_parser('evaluate', help="Evaluate the learned segmenter/standardizer models on the given test files")
    eval_parser.add_argument('what', choices=['segmenter', 'standardizer'], help="What do you want to evaluate? (Segmenter/Standardizer)")
    eval_parser.add_argument('-l', '--model', required=True, help="Path to model file")
    eval_parser.add_argument('-f', '--files', nargs='+', help='List of all files to be evaluated')
    eval_parser.set_defaults(func=evaluate)

    options = parser.parse_args()
    options.func(options)


if __name__ == "__main__":
    main()
