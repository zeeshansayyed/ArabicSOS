import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

from scripts.extractor import OldFeatureExtractor, SegmenterLE, StandardizerLE, StandardizerDatasetHelper, SegmenterDatasetHelper
from scripts.util import get_num_lines, clean_arabic

import pickle


class Dataset(ABC):

    def __init__(self, type, template, label_style, train_file, dev_file, test_file, name):
        self.type = type
        self.name = name
        self.template = template
        self.fe = OldFeatureExtractor(self.template)
        self.label_style = label_style
        if self.type == 'segmenter':
            self.le = SegmenterLE(self.label_style)
            self.helper = SegmenterDatasetHelper(self.label_style)
        elif self.type == 'standardizer':
            self.le = StandardizerLE(self.label_style)
            self.helper = StandardizerDatasetHelper(self.label_style)
        if train_file:
            print("Creating train set...")
            self.Xtrain, self.ytrain = self.create_dataset(train_file)
        if dev_file:
            print("Creating dev set...")
            self.Xdev, self.ydev = self.create_dataset(dev_file)
        else:
            self.Xdev, self.ydev = None, None
        if test_file:
            print("Creating test set...")
            self.Xtest, self.ytest = self.create_dataset(test_file)
        else:
            self.Xtest, self.ytest = None, None

    def __str__(self):
        return "Type: {}, Name: {}, Template: {}, Label Style: {}".format(self.type, self.name, self.template, self.label_style)

    @abstractmethod
    def extract(self, line):
        pass

    def create_dataset(self, filename):
        X, y = [], []
        with open(filename, 'r') as infile:
            for line in tqdm(infile, total=get_num_lines(filename)):
                features, labels = self.extract(line)
                X += features
                y += labels
        return np.array(X), np.array(y)


class SegmenterDataset(Dataset):
    """
    The constructor takes the segmented file as input.
    It will automatically extract the features for characters
    according to the specified template and labels for the corresponding
    characters according to the specified style and create number arrays for X and y
    """

    def __init__(self, template, train_file, dev_file, test_file=None, name=None, label_style='binary_plus'):
        super().__init__('segmenter', template, label_style, train_file, dev_file, test_file, name)
        self.le = SegmenterLE()

    def extract(self, line):
        target_line = self.helper.get_target_sequence(line)
        input_line = self.helper.get_input_sequence(target_line)
        no_space_flags = self.helper.get_nonspace_flags(input_line)
        sent_features = self.fe.sent_to_features(input_line, no_space_flags)
        sent_labels = self.le.sent_to_labels(target_line.split())
        return sent_features, sent_labels


class StandardizerDataset(Dataset):
    """SSO stands for Sub Standard Orthography"""

    def __init__(self, template, train_file, dev_file=None, test_file=None, name=None, label_style='eight_class'):
        self.substandard_dict = {
            'أ': 'ا',
            'إ': 'ا',
            'آ': 'ا',
            'ة': 'ه'
        }
        super().__init__('standardizer', template, label_style, train_file, dev_file, test_file, name)

    def _extract(self, line):
        target_line = self.helper.get_target_sequence(line)
        input_line = self.helper.get_input_sequence(target_line)
        target_sequence = target_line.split()
        input_sequence = input_line.split()
        features, labels = [], []
        char_pos, word_pos = -1, 0
        for char_no, char in enumerate(input_line):
            if char == ' ':
                char_pos = -1
                word_pos += 1
            else:
                char_pos += 1

            if self.helper.should_standardize(char_no, input_line):
                features.append(self.fe.char_to_features(char_pos, word_pos, input_sequence))
                labels.append(self.le.char_to_label(target_sequence[word_pos][char_pos]))

        return features, labels

    def extract(self, line):
        target_line = self.helper.get_target_sequence(line)
        input_line = self.helper.get_input_sequence(target_line)
        flags = self.helper.get_standardizable_flags(input_line)
        features = self.fe.sent_to_features(input_line, flags)
        labels = self.le.sent_to_labels(target_line, flags)
        return features, labels


if __name__ == "__main__":
    d1 = SegmenterDataset('t1', 'data/segmenter/train4.txt', 'data/segmenter/dev1.txt')
    pickle.dump(d1, open('data/segmenter/seg2.ds', 'wb'))
    # d2 = StandardizerDataset('t1', 'data/standardizer/siyar.txt', 'data/standardizer/albidya_walnihaya.txt')
    # pickle.dump(d2, open('data/standardizer/siyar2.ds', 'wb'))
    # d2 = pickle.load(open('data/standardizer/std.ds','rb'))
    print(d1.Xtrain.shape)