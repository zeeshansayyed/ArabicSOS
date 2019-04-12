#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 28 2:19pm 2019

@author: Zeeshan Ali Sayyed
"""

import mmap
from scripts.preprocessor import clean_arabic

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

substandard_dict = {
    'أ': 'ا',
    'إ': 'ا',
    'آ': 'ا',
    'ة': 'ه'
}

def substandardize_word(word):
    word = list(word)
    for pos, char in enumerate(word):
        if char in substandard_dict:
            word[pos] = substandard_dict[char]
    if word[-1] == 'ي':
        word[-1] = 'ى'

    return ''.join(word)

def substandardize_line(line):
    line = clean_arabic(line).split()
    sso_line = []
    for word in line:
        sso_line.append(substandardize_word(word))
    return ' '.join(sso_line)
