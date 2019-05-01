from abc import ABC, abstractmethod
import csv
from scripts.util import clean_arabic
from scripts.feature_extractor import char_to_features as c2f
from scripts.config import feature_templates
import itertools


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

    def _sent_to_features(self, sentence, flat=False):
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

    def sent_to_features(self, sentence, char_flags=None):
        """Features will only be extracted for flags which are > 0"""
        sent_features = []
        split_sentence = sentence.split()
        if char_flags is None:
            char_flags = [1] * len(sentence)

        char_pos, word_pos = -1, 0  # Used to index characters in `substandard_line`
        for char_no, char in enumerate(sentence):  # Used to index characters in `result`
            if char == ' ':
                char_pos = -1
                word_pos += 1
            else:
                char_pos += 1

            if char_flags[char_no] > 0:
                try:
                    sent_features.append(self.char_to_features(char_pos, word_pos, split_sentence))
                except:
                    print(sentence)
                    print(split_sentence)
                    print(char_flags)
                    print(char_pos, word_pos)
                    raise Exception()

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
    """
    This class will provide the following functionality:
    1.  Extracted input and output lines from annotated lines. Feature can then
        be extracted from input lines and labels from output lines to create
        a dataset for the machine learning model. This will also handle the
        cleaning of the sentences
    2. Extract numeric labels from the output lines
    3. Recreate output form from numeric labels which are predicted by the model.

    Note: Each subclass will have to support different styles of annotation.
    """

    def __init__(self, style):
        self.style = style

    def char_to_label(self, char):
        pass

    def word_to_labels(self, word):
        return list(map(self.char_to_label, word))

    def sent_to_labels(self, sent):
        # print(sent, type(sent))
        if type(sent) is str:
            # print(type(sent))
            sent = sent.split()
        return list(itertools.chain(*map(self.word_to_labels, sent)))

    def file_to_labels(self, infile):
        file_labels = []
        with open(infile, 'r') as infile:
            line = clean_arabic(line)
            file_labels += self.sent_to_labels(line)
        return file_labels

    def labels_to_word(self, word, labels):
        pass



class SegmenterLE(LabelExtractor):
    """
    Styles supported:
    1. Binary Plus: Segmented are separated by a '+' and labels are binary (i.e. 0 and 1)
    """
    def __init__(self, style='binary_plus'):
        super().__init__(style)

    def word_to_labels(self, word):
        # print(word)
        if self.style == 'binary_plus':
            word_split = word.split('+')
            word_len = sum([len(i) for i in word_split])
            labels = [0] * word_len
            curr_ind = -1
            for segment in word_split:
                curr_ind += len(segment)
                labels[curr_ind] = 1
            return labels
        else:
            raise Exception("Style: {} is not supported".format(self.style))

    def labels_to_word(self, word, labels):
        # print(word, labels)
        segmented_word = []
        if self.style == 'binary_plus':
            for i, label in enumerate(labels):
                segmented_word.append(word[i])
                if label == 1:
                    segmented_word.append('+')
            if segmented_word[-1] == '+':
                segmented_word.pop()
            return ''.join(segmented_word)
        else:
            raise Exception("Style: {} is not supported".format(self.style))

    def labels_to_sent(self, sentence, labels):
        # print(sentence)
        # print(len(sentence), len(labels))
        result = []
        word_lengths = map(len, sentence.split())
        word_start = 0
        label_start = 0
        for word_length in word_lengths:
            word_end = word_start + word_length
            label_end = label_start + word_length
            word = sentence[word_start:word_end]
            word_labels = labels[label_start:label_end]
            # print(word, word_labels)
            result.append(self.labels_to_word(word, word_labels))
            word_start = word_end + 1
            label_start = label_end
        # print(word_end, label_end)
        return ' '.join(result)


class StandardizerLE(LabelExtractor):
    """
    Styles supported:
    1. Eight class: Three alphabets (ا, ه, ى) with 8 SSO mappings
    """
    def __init__(self, style='eight_class'):
        super().__init__(style)

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
        else:
            raise Exception("Style: {} is not supported".format(self.style))

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
        else:
            raise Exception("Style: {} is not supported".format(self.style))

    def sent_to_labels(self, sent, char_flags=None):
        if char_flags is None:
            char_flags = [1] * len(sent)
        return [self.char_to_label(char) for char, flag in zip(sent, char_flags) if flag > 0]

    def labels_to_sent(self, sentence, labels, flags):
        result = list(sentence)
        for index, (label, flag) in enumerate(zip(labels, flags)):
            if flag > 0:
                result[index] = self.label_to_char(label)
        return ''.join(result)


class DatasetHelper(ABC):

    def __init__(self, style):
        self.style = style

    def _style_not_supported(self):
        raise Exception("Style: {} is not supported".format(self.style))

    @abstractmethod
    def get_target_sequence(self, annotated_line):
        pass

    @abstractmethod
    def get_input_sequence(self, target_sequence):
        pass

    def get_nonspace_flags(self, sentence):
        return [0 if char == ' ' else 1 for char in sentence]


class SegmenterDatasetHelper(DatasetHelper):

    def __init__(self, style):
        super().__init__(style)

    def get_target_sequence(self, annotated_line):
        if self.style == 'binary_plus':
            clean_line = clean_arabic(annotated_line, contains_plus=True)
            return ' '.join(clean_line.split())
        else:
            super()._style_not_supported()

    def get_input_sequence(self, target_sequence):
        if self.style == 'binary_plus':
            return target_sequence.replace('+', '')
        else:
            super()._style_not_supported()


class StandardizerDatasetHelper(DatasetHelper):

    def __init__(self, style):
        super().__init__(style)
        self.eight_class_dict = {
            'أ': 'ا',
            'إ': 'ا',
            'آ': 'ا',
            'ة': 'ه'
        }
        self.eight_class_set = set(['ا', 'ه'])
        self.eight_class_word_end = set(['ى'])

    def get_target_sequence(self, annotated_line):
        if self.style == 'eight_class':
            clean_line = clean_arabic(annotated_line)
            return ' '.join(clean_line.split())
        else:
            super()._style_not_supported()

    def get_input_sequence(self, target_sequence):
        if len(target_sequence) == 0:
            return ''
        if self.style == 'eight_class':
            target_sequence = list(target_sequence)

            # Loop through all characters except the last
            for char_no, char in enumerate(target_sequence[:-1]):
                if char in self.eight_class_dict:
                    target_sequence[char_no] = self.eight_class_dict[char]
                if char == 'ي' and target_sequence[char_no + 1] == ' ':
                    target_sequence[char_no] = 'ى'

            # print(target_sequence)
            # Handling the last character
            char = target_sequence[-1]
            if char in self.eight_class_dict:
                target_sequence[-1] = self.eight_class_dict[char]
            elif target_sequence[-1] == 'ي':
                target_sequence[-1] = 'ى'

            return ''.join(target_sequence)
        else:
            super()._style_not_supported()

    def should_standardize(self, char_no, sentence):
        char = sentence[char_no]
        if self.style == 'eight_class':
            if char in self.eight_class_set:
                return True
            elif char in self.eight_class_word_end:
                if char_no == len(sentence) - 1:
                    return True
                elif sentence[char_no + 1] == ' ':
                    return True
                else:
                    return False
            else:
                return False
        else:
            super()._style_not_supported()

    def get_standardizable_flags(self, sentence):
        return [1 if self.should_standardize(char_no, sentence) else 0 for char_no in range(len(sentence))]

    def get_substandardizable_indices(self, sentence):
        pass




if __name__ == "__main__":
    fe = OldFeatureExtractor('t1')
    le = SegmenterLE('binary_plus')
    helper = SegmenterDatasetHelper('binary_plus')
    s = 'ال+إنساني+ة ال+حق+ة .'
    target_sequence = helper.get_target_sequence(s)

    print(s)
    print(helper.get_target_sequence(s))
    # print(fe.sent_to_features(s, flat=True))
    # print(fe.char_to_features(0, 0, s))
    # print(fe.word_to_features(0, s.split()))
    # print(fe.file_to_features('sample/P105small', 'sample/P105small.feat'))
    # fe.sent_to_features(s.split())
    # print(le.sent_to_labels(s.split()))
