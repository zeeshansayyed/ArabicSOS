#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:59:23 2018

@author: Zeeshan Ali Sayyed
"""

from scripts.util import clean_arabic
import csv
import pandas as pd


def _get_single_char(word, curr_pos, direction, distance, fill_char='*'):
    if direction == 'left':
        target_pos = curr_pos - distance
    else:
        target_pos = curr_pos + distance

    if target_pos < 0 or target_pos >= len(word):
        return fill_char
    else:
        return word[target_pos]


def _get_multiple_chars(fcode, word, curr_pos):
    if fcode.startswith('next'):
        n_chars = int(fcode.replace('next', '').replace('letters', ''))
        direction = 'right'
        distance = 0
    elif fcode.startswith('prev'):
        n_chars = int(fcode.replace('prev', '').replace('letters', ''))
        direction = 'left'
        distance = 0
    else:
        raise Exception("Feature code ending in letters should begin with next or prev")
    feature = ""
    for i in range(n_chars, 0, -1):
        feature += _get_single_char(word, curr_pos, direction, distance + i, fill_char='#')
        # print(i, feature)
    if direction == 'left':
        return feature
    else:
        return feature[::-1]


def _get_affix(fcode, word_pos, sentence, prefix_length=3):
    fcode = fcode.split('_')
    if fcode[0] == 'prev':
        word_pos -= 1
    elif fcode[0] == 'following':
        word_pos += 1
    if word_pos < 0 or word_pos >= len(sentence):
        return '#'

    word = sentence[word_pos]

    if fcode[2] == 'prefix':
        return word[:prefix_length]
    elif fcode[2] == 'suffix':
        return word[-prefix_length:]
    else:
        raise Exception("Unsupported feature code: {}".format(fcode))


def _process_feature_code(char_pos, word_pos, sentence, fcode):
    # print(fcode)
    if fcode.startswith('minus'):
        direction = 'left'
        steps = int(fcode[5:])
        feature = _get_single_char(sentence[word_pos], char_pos, direction, steps)
    elif fcode.startswith('plus'):
        direction = 'right'
        steps = int(fcode[4:])
        feature = _get_single_char(sentence[word_pos], char_pos, direction, steps)
    elif fcode == 'focus':
        feature = sentence[word_pos][char_pos]
    elif fcode.endswith('letters'):
        feature = _get_multiple_chars(fcode, sentence[word_pos], char_pos)
    elif fcode.endswith('fix'):
        feature = _get_affix(fcode, word_pos, sentence)
    elif fcode == 'chr_position':
        feature = char_pos
    else:
        feature = None

    if feature != None:
        return feature
    else:
        raise Exception("Unsupported feature code: {}".format(fcode))


def char_to_features(char_pos, word_pos, sentence, feature_codes):
    char_features = []
    for fcode in feature_codes:
        char_features.append(_process_feature_code(char_pos, word_pos, sentence, fcode))
    # print(char_features)
    return char_features


def word_to_features(word_pos, sentence, feature_codes):
    word_features = []
    for char_pos in range(len(sentence[word_pos])):
        char_features = []
        for fcode in feature_codes:
            try:
                char_features.append(_process_feature_code(char_pos, word_pos, sentence, fcode))
            except:
                print("Error in word_pos={} (word={}), char={} and fcode={}".format(word_pos,
                                                                                    sentence[word_pos],
                                                                                    sentence[word_pos][char_pos],
                                                                                    fcode))
                raise ("Error in creating features")
        word_features.append(char_features)
    return word_features


def sent_to_features(sentence, feature_codes):
    sentence = clean_arabic(sentence).split()
    sent_features = []
    for word_pos in range(len(sentence)):
        sent_features.append(word_to_features(word_pos, sentence, feature_codes))
    return sent_features


def file_to_features(filepath, feature_codes):
    """
    TODO: Line, word and character numbering starts from 1 in order to consistent with existing trained models. Change them to start from 0 in future models and remove this Warning!
    """
    infile = open(filepath)
    file_features = []
    for line_no, line in enumerate(infile):
        line_features = sent_to_features(line, feature_codes)
        for word_no, word_features in enumerate(line_features):
            for char_no, char_features in enumerate(word_features):
                file_features.append([line_no + 1, word_no + 1, char_no + 1] + char_features)
    return file_features


def word_to_labels(word, scheme="default"):
    word_split = word.split('+')
    word_len = sum([len(i) for i in word_split])
    if scheme == "default":
        labels = [0] * word_len
        curr_ind = -1
        for segment in word_split:
            curr_ind += len(segment)
            labels[curr_ind] = 1
    return labels


def sent_to_labels(sentence, scheme="default"):
    words = sentence.split()
    labels = []
    for word in words:
        labels.append(word_to_labels(word, scheme))
    return labels


def extract_features(filepath, feature_codes, write_style='csv', out_file_path=None, return_style=None):
    """
    Extracts features from the given file name. Currently supported file format is CSV.
    feature_codes is the list indicating the features which should be extracted.
    If out_file_name is present, the file will be written at that location, else it will be ignored.
    """
    infile = open(filepath)
    lines = infile.readlines()
    infile.close()

    line_features = []
    for line in lines:
        line_features.append(sent_to_features(line, feature_codes))
    file_features = file_to_features(filepath, feature_codes)

    if out_file_path:
        if write_style == 'csv':
            with open(out_file_path, 'w') as out_file:
                fwriter = csv.writer(out_file, delimiter='\t')
                fwriter.writerow(['line_no', 'word_no', 'chr_position'] + feature_codes)
                fwriter.writerows(file_features)
        else:
            raise Exception("Other writing styles aren't supported yet!")

    if return_style:
        if return_style == 'raw':
            return file_features
        elif return_style == 'dataframe':
            data = []
            columns = ['line_no', 'word_no', 'chr_position'] + feature_codes
            feature_frame = pd.DataFrame(file_features, columns=columns)
            return feature_frame
        else:
            raise Exception("Other return styles aren't supported yet!")


if __name__ == "__main__":
    print("Feature Extractor called as independent script")
    feature_codes = ['minus5', 'minus4', 'minus3', 'minus2', 'minus1', 'focus',
                     'plus1', 'plus2', 'plus3', 'plus4', 'plus5', 'next2letters',
                     'prev2letters', 'prev_word_suffix', 'following_word_prefix',
                     'focus_word_prefix', 'focus_word_suffix']
    sentence = "الكاتب: محمد رشيد رضا"
    # f = sent_to_features(sentence, feature_codes)
    # print(len(f), len(f[0]), len(f[0][0]))
    f = extract_features('../data/playground/TestFile', feature_codes, return_style='dataframe')
    print(f)
