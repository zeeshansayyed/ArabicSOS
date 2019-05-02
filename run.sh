#!/usr/bin/env bash
template=t3
name="segmenter_"$template"_test"
python arabicsos.py train segmenter --train data/segmenter/train4.txt --dev data/segmenter/dev1.txt --name $name -p $template --algorithm catboost
python arabicsos.py evaluate segmenter --files data/segmenter/dev1.txt data/segmenter/test1.txt data/segmenter/test2.txt --model $name.mod
