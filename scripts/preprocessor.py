#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 2018

seprate punctuation
remove diacritics
seprate digits from words

@author: Dr. Mohamed Emad
Modified by: Zeeshan Ali Sayyed
"""

line = "»«لدينا900دارس؟"

damma = "ُ"
sukun = "ْ"
fatha = "َ"
kasra = "ِ"
shadda = "ّ"
tanweendam = "ٌ"
tanweenfath = "ً"
tanweenkasr = "ٍ"
tatweel = "ـ"

tashkil = (damma, sukun, fatha, kasra, shadda, tanweendam, tanweenfath, tanweenkasr, tatweel)


def removeTashkil(word):
    w = [letter for letter in word if letter not in tashkil]
    return "".join(w)


import re


def sepDigits(someString):
    return " ".join(re.split('(\d+)', someString))


def sepPunc(someString):
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~؛،؟؛.»«”'
    out = []
    for char in someString:
        if char in punc:
            out.append(' ' + char + ' ')
        else:
            out.append(char)
    return ''.join(out)


def clean_arabic(someString):
    return sepPunc(sepDigits(removeTashkil(someString)))


# main
import sys

if __name__ == '__main__':
    try:
        infile = open(sys.argv[1])
        for line in infile:
            line = line.strip().replace('_', '-')  # the '_' char is special
            # if cleanArabic(line) == "":
            #	print("_______________")
            # elif not line:
            #	print("GGGG")
            # else:
            print(cleanArabic(line))
    except IndexError:
        print("There seems to be something wrong. You need a file to clean!!!")
