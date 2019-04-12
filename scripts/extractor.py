from abc import ABC, abstractmethod
import csv
from scripts.preprocessor import clean_arabic
from scripts.feature_extractor import char_to_features as c2f
from scripts.config import feature_templates
import itertools
from pprint import pprint
# import pudb;
# pu.db

class FeatureExtractor(ABC):

    def __init__(self, template):
        self.template = feature_templates[template]

    @abstractmethod
    def char_to_features(self, char_pos, word_pos, sentence):
        pass

    def word_to_features(self, word_pos, sentence):
        word_features = []
        for char_pos in range(len(sentence[word_pos])):
            word_features.append(self.char_to_features(char_pos, word_pos, sentence))
        return word_features

    def sent_to_features(self, sentence, flat=False):
        if type(sentence) == str:
            sentence = sentence.split()
        sent_features = []
        for word_pos in range(len(sentence)):
            word_features = self.word_to_features(word_pos, sentence)
            if flat:
                for char_features in word_features:
                    sent_features.append(char_features)
            else:
                sent_features.append(word_features)
        return sent_features

    def file_to_features(self, infile, outfile=None, flat=False):
        """
        infile:     Input file from which features should be generated
        outfile:    Output file to which features should be written. If no outfile 
                    is specified, then the featues are returned in list form.
        flatten:    If true, then the returned feature list is a list(list) char(features)
                    Spaces are ommitted and features aren't generated for it.
                    If false, it is list(list(list(list))) sentences(word(char(features)))
        """
        if outfile:
            outfile = open(outfile, 'w')
            fwriter = csv.writer(outfile, delimiter='\t')
            fwriter.writerow(self.template)
        else:
            file_features = []
        
        infile = open(infile, 'r')
        for line in infile:
            line = clean_arabic(line)
            if not outfile and not flat:
                sent_features = self.sent_to_features(line)
                file_features.append(sent_features)
            else:
                sent_features = self.sent_to_features(line, flat=True)
                if outfile:
                    fwriter.writerows(sent_features)
                else:
                    for char_features in sent_features:
                        file_features.append(char_features)
        infile.close()

        if outfile:
            outfile.close()
        else:
            return file_features

class OldFeatureExtractor(FeatureExtractor):

    def __init__(self, template):
        super().__init__(template)
    
    def char_to_features(self, char_pos, word_pos, sentence):
        # print(self.template)
        return c2f(char_pos, word_pos, sentence, self.template)


class LabelExtractor():

    def __init__(self, style):
        self.style = style

    def char_to_label(self, char):
        pass

    def word_to_labels(self, word):
        return list(map(self.char_to_label, word))

    def labels_to_word(self, word, labels):
        pass

    def sent_to_labels(self, sent):
        if type(sent) == 'str':
            sent = sent.split()
        return list(itertools.chain(*map(self.word_to_labels, sent)))

    def file_to_labels(self, infile):
        file_labels = []
        with open(infile, 'r') as infile:
            line = clean_arabic(line)
            file_labels += self.sent_to_labels(line)
        return file_labels

class SegmenterLE(LabelExtractor):

    def __init__(self, style='binary_plus'):
        super().__init__(style)

    def word_to_labels(self, word):
        if self.style == 'binary_plus':
            word_split = word.split('+')
            word_len = sum([len(i) for i in word_split])
            labels = [0] * word_len
            curr_ind = -1
            for segment in word_split:
                curr_ind += len(segment)
                labels[curr_ind] = 1
            return labels

    def labels_to_word(self, word, labels):
        segmented_word = ""
        if self.style == 'binary_plus':
            for i in range(len(word)):
                segmented_word += word[i]
                if labels[i] and i < len(word)-1:
                    segmented_word += '+'
            return segmented_word


class StandardizerLE(LabelExtractor):

    def __init__(self, style='eight_class'):
        super().__init__(style)
        self.alif_ha = set(('ا', 'ه'))
        self.alif_maksura = 'ى'

    def char_to_label(self, char):
        if self.style == 'eight_class':
            return {
                'ا': 0,
                'أ': 1,
                'إ': 2,
                'آ': 3,
                'ة': 4,
                'ه': 5,
                'ي': 6,
                'ى': 7
            }[char]

    def label_to_char(self, label):
        if self.style == 'eight_class':
            return {
                0: 'ا',
                1: 'أ',
                2: 'إ',
                3: 'آ',
                4: 'ة',
                5: 'ه',
                6: 'ي',
                7: 'ى'
            }[label]

    def should_standardize(self, char_no, word_no, sentence):
        word = sentence[word_no]
        char = word[char_no]
        if char in self.alif_ha or (char == 'ى' and char_no == len(word) - 1):
            return True
        else:
            return False


if __name__ == "__main__":
    fe = OldFeatureExtractor('t1')
    le = SegmenterLE('binary_plus')
    s = "Zee+shan Al+i Say+yed"
    # print(fe.sent_to_features(s, flat=True))
    # print(fe.char_to_features(0, 0, s))
    # print(fe.word_to_features(0, s.split()))
    # print(fe.file_to_features('sample/P105small', 'sample/P105small.feat'))
    # fe.sent_to_features(s.split())
    print(le.sent_to_labels(s.split()))
