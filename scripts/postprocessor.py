#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 13:53:52 2018

@author: shahensha
"""

from scripts.preprocessor import clean_arabic


def _humanify_word(word, markings):
    readable_word = ""
    for i in range(len(word)):
        readable_word += word[i]
        if markings[i] and i < len(word)-1:
            readable_word += '+'
    return readable_word


def _humanify_sentence(sentence, markings_list):
    sentence = clean_arabic(sentence).split()
    readable_sentence = ""
    for i in range(len(sentence)):
        readable_sentence += _humanify_word(sentence[i], markings_list[i]) + ' '
    return readable_sentence.strip()


def get_human_readable_segmentation(sentences, all_markings):
    readable_sentences = []
    if type(all_markings) == "list":
        for sentence, markings_list in zip(sentences, all_markings):
            readable_sentences.append(_humanify_sentence(sentence, markings_list))
    else:
        for i, sentence in enumerate(sentences):
            readable_sentences.append(_humanify_sentence(sentence, list(all_markings[i + 1])))
    return readable_sentences
