import numpy as np
from abc import ABC, abstractmethod
from scripts.preprocessor import clean_arabic
from scripts.extractor import OldFeatureExtractor, StandardizerLE, SegmenterLE
from scripts.util import substandardize_line

import pickle


class Actor(ABC):
    """ This is a super class for all the elements in the NLP Pipeline such as 
    Segmenter, Standardizer, POS Tagger and NER """

    def __init__(self, model, feature_extractor, label_extractor):
        self.model = model
        self.fe = feature_extractor
        self.le = label_extractor

    def act_on_char(self, char_no, word_no, line, **kwargs):
        pass

    def act_on_word(self, word_no, line, **kwargs):
        orig_word = line[word_no]
        # print(orig_word)
        res_word = []
        for char_no in range(len(orig_word)):
            res_word.append(self.act_on_char(char_no, word_no, line, **kwargs))
        # print(res_word)
        return ''.join(res_word)
        
    def act_on_line(self, line, **kwargs):
        orig_line = clean_arabic(line).split()
        res_line = []
        for word_no in range(len(orig_line)):
            res_line.append(self.act_on_word(word_no, orig_line, **kwargs))
        return " ".join(res_line)

    def act_on_file(self, infile, outfile):
        with open(infile, 'r', encoding='utf-8') as infile, open(outfile, 'w', encoding='utf-8') as outfile:
            for line in infile:
                print(self.act_on_line(line), file=outfile)


class Segmenter(Actor):
    """This class implements the segmentation functionality. It takes in the name of the model to load,
    the feature extactor and a label extractor in its constructor. Additionally, if you also pass in
    an object of Standardizer to the constructor, it will standardize the text before segmenting it"""

    def __init__(self, model, feature_extractor, label_extractor, standardizer=None):
        super().__init__(model, feature_extractor, label_extractor)
        self.standardizer = standardizer

    def act_on_word(self, word_no, line, **kwargs):
        word_features = np.array(self.fe.word_to_features(word_no, line))
        labels = self.model.predict(word_features)
        return self.le.labels_to_word(line[word_no], labels)

    def act_on_line(self, line):
        if self.standardizer:
            line = self.standardizer.act_on_line(line)
        return super().act_on_line(line)

class Standardizer(Actor):
    """This class implements the standardization functionality. It takes in a model, feature extractor
    and a label extractor. After creating the object, the following two functions can be called:
    1) act_on_line() and (2) act_on_file()
    The input sentence is first substandardized and features are extracted from this sentence. We then
    loop through the input sentence and standardize the necessary characters: """

    def __init__(self, model, feature_extractor, label_extractor):
        super().__init__(model, feature_extractor, label_extractor)

    def act_on_char(self, char_no, word_no, line, **kwargs):
        substandard_line = kwargs['substandard_line']
        if self.le.should_standardize(char_no, word_no, line):
            char_features = self.fe.char_to_features(char_no, word_no, substandard_line)
            char_features = np.array(char_features).reshape(1, len(self.fe.template))
            label = self.model.predict(char_features)[0]
            return self.le.label_to_char(label)
        else:
            return line[word_no][char_no]

    def act_on_line(self, line):
        substandard_line = substandardize_line(line)
        assert len(line) == len(substandard_line)
        substandard_line = substandard_line.split()
        return super().act_on_line(line, substandard_line=substandard_line)



if __name__ == "__main__":
    # s = "مغاني الشعب طيبا في المغاني بمنزلة الربيع من الزمان"
    # sso = substandardize_line(s)
    # std_model = pickle.load(open('models/lgbm_standardizer.mod', 'rb'))
    seg_model = pickle.load(open('models/default_segmenter.mod', 'rb'))
    # std_le = StandardizerLE()
    seg_le = SegmenterLE()
    fe = OldFeatureExtractor('t1')
    # std = Standardizer(std_model, fe, std_le)
    seg = Segmenter(seg_model, fe, seg_le)
    # print(s)
    # print(sso)
    # print(std.act_on_line(s))
    # print(seg.act_on_line(sso))
    seg.act_on_file('sample/seg.txt', 'sample/seg.seg')
    # std.act_on_file('sample/small_siyar.sso', 'sample/small_siyar.std')
        