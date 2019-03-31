#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 28 2:19pm 2019

@author: Zeeshan Ali Sayyed
"""

import mmap

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines