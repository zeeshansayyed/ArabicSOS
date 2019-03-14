#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 11:27:37 2018

@author: Zeeshan Ali Sayyed
TEST: To see whether Pycharm syncs correctly.
"""
import argparse
from scripts.feature_extractor import extract_features
from catboost import CatBoostClassifier
from scripts.postprocessor import get_human_readable_segmentation

feature_codes_1 = ['minus5', 'minus4', 'minus3', 'minus2', 'minus1', 'focus',
                   'plus1', 'plus2', 'plus3', 'plus4', 'plus5', 'next2letters',
                   'prev2letters', 'prev_word_suffix', 'following_word_prefix',
                   'focus_word_prefix', 'focus_word_suffix']




def segment(infile, outfile):
    model = CatBoostClassifier()
    model.load_model("models/catboost_1.model")
    ff = extract_features(infile, feature_codes_1, return_style='dataframe')
    X = ff[['chr_position'] + feature_codes_1]
    y = model.predict(X)
    ff['predictions'] = y.astype(int)
    preds = ff.groupby(['line_no', 'word_no'])['predictions'].apply(list)
    with open(infile, 'r') as reader:
        sentences = reader.readlines()
    seg_sents = get_human_readable_segmentation(sentences, preds)
    with open(outfile, 'w') as segfile:
        segfile.writelines('\n'.join(seg_sents))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arabic Segmenter and Orthography Standardizer")
    parser.add_argument('in_file', metavar='I', help="Path to the input file which needs to be segmented")
    parser.add_argument("-o", "--out-file", help="Path of the out_file. If it is not provideed, it will be stored in the same directory as the input file.")
    parser.add_argument("-v", "-verbose", action="store_true")
    args = parser.parse_args()

    if not args.out_file:
        args.out_file = args.in_file + '.segmented'
    segment(args.in_file, args.out_file)
