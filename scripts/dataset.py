import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

from scripts.extractor import OldFeatureExtractor, SegmenterLE, StandardizerLE
from scripts.preprocessor import clean_arabic
from scripts.util import get_num_lines



class Dataset(ABC):

    def __init__(self, type, template, label_style, train_file, dev_file, test_file, name):
        self.type = type
        self.name = name
        self.template = template
        self.fe = OldFeatureExtractor(self.template)
        self.label_style = label_style
        if self.type == 'segmenter':
            self.le = SegmenterLE(label_style)
        elif self.type == 'standardizer':
            self.le = StandardizerLE(label_style)
        if train_file:
            print("Creating train set...")
            self.Xtrain, self.ytrain = self.create_dataset(train_file)
        if dev_file:
            print("Creating dev set...")
            self.Xdev, self.ydev = self.create_dataset(dev_file)
        if test_file:
            print("Creating test set...")
            self.Xtest, self.ytest = self.create_dataset(test_file)

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
        segline = clean_arabic(line, contains_plus=True).strip()
        rawline = segline.replace('+', '')
        sent_features = self.fe.sent_to_features(rawline.split(), flat=True)
        sent_labels = self.le.sent_to_labels(segline.split())
        return sent_features, sent_labels


class StandardizerDataset(Dataset):
    """SSO stands for Sub Standard Orthography"""

    def __init__(self, template, train_file, dev_file, test_file=None, name=None, label_style='eight_class'):
        self.substandard_dict = {
            'أ': 'ا',
            'إ': 'ا',
            'آ': 'ا',
            'ة': 'ه'
        }
        self.substandard_chars = set(('ا', 'ه'))
        super().__init__('standardizer', template, label_style, train_file, dev_file, test_file, name)

    def substandardize(self, word):
        word = list(word)
        for pos, char in enumerate(word):
            if char in self.substandard_dict:
                word[pos] = self.substandard_dict[char]
        if word[-1] == 'ي':
            word[-1] = 'ى'

        return ''.join(word)

    def extract(self, line):
        stdline = clean_arabic(line).strip().split()
        ssoline = [self.substandardize(w) for w in stdline]
        features, labels = [], []
        for word_pos, word in enumerate(ssoline):
            for char_pos, char in enumerate(word):
                if char in self.substandard_chars or (char_pos == len(word)-1 and char == 'ى'):
                    features.append(self.fe.char_to_features(char_pos, word_pos, ssoline))
                    labels.append(self.le.char_to_label(stdline[word_pos][char_pos]))
        return features, labels


if __name__ == "__main__":
    d1 = SegmenterDataset('t1', 'data/segmenter/train4.txt', 'data/segmenter/dev1.txt')
    print(d1)
    # d2 = StandardizerDataset('t1', 'data/standardizer/siyar.txt', 'data/standardizer/siyar.txt')