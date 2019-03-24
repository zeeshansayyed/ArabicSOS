#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed March 23 11:27:37 2019

@author: Zeeshan Ali Sayyed
"""

import argparse

def train(options):
    print("Called Train")
    print(options)

def segment(options):
    print("Called Segment")
    print(options)

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
    train_parser.add_argument('-t', '--train', required=True, help='train location')
    train_parser.add_argument('-d', '--dev', required=True, help='dev location')
    train_parser.add_argument('-s', '--test', help='test location')
    train_parser.add_argument('-l', '--model', type=argparse.FileType('w'), help='Name of the output model file')
    train_parser.set_defaults(func=train)

    # Subparser for segment command
    segment_parser = subparsers.add_parser('segment', help="Run the segmenter")
    # segment_parser.add_argument('--standardize', help="Pass the model that will be used to standardize the text before segmenting. If no model is passed, the default standardizer model will be used")
    segment_parser.add_argument('-i', '--input', type=argparse.FileType('r'), help="Input file")
    segment_parser.add_argument('-o', '--output', type=argparse.FileType('w'), help="Output file. If absent, .seg extension will be used")
    segment_parser.add_argument('-s', '--standardize', action='store_true', help="Standardize the file before segmenting")
    segment_parser.add_argument('-m', '--mode', choices=['interactive', 'batch', 'web'])
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