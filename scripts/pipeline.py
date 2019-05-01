import numpy as np
from abc import ABC, abstractmethod
import pickle
from tqdm import tqdm
from collections import Counter

from scripts.util import clean_arabic, get_num_lines
from scripts.extractor import OldFeatureExtractor, StandardizerLE, SegmenterLE, SegmenterDatasetHelper, StandardizerDatasetHelper
from scripts.util import substandardize_line
from numba import jit



class Actor(ABC):
    """ This is a super class for all the elements in the NLP Pipeline such as 
    Segmenter, Standardizer, POS Tagger and NER """

    def __init__(self, model, feature_extractor, label_extractor, dataset_helper):
        self.model = model
        self.fe = feature_extractor
        self.le = label_extractor
        self.helper = dataset_helper

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
        # orig_line = clean_arabic(line).split()
        orig_line = line.split()
        res_line = []
        for word_no in range(len(orig_line)):
            res_line.append(self.act_on_word(word_no, orig_line, **kwargs))
        return " ".join(res_line)

    def act_on_file(self, infile, outfile):
        num_lines = get_num_lines(infile)
        with open(infile, 'r', encoding='utf-8') as infile, open(outfile, 'w', encoding='utf-8') as outfile:
            for line in tqdm(infile, total=num_lines):
                print(self.act_on_line(line), file=outfile)

    def get_character_errors(self, target_sequence, predicted_sequence):
        """
        Should be overwridden by the subclass
        :param target_sequence:
        :param predicted_sequence:
        :return: Counter() containing total characters and error characters
        """
        return Counter()

    def get_word_errors(self, target_sequence, predicted_sequence):
        errors = []
        target_sequence = target_sequence.split()
        predicted_sequence = predicted_sequence.split()
        # print(target_sequence, predicted_sequence)
        assert len(target_sequence) == len(predicted_sequence)
        for target, predicted in zip(target_sequence, predicted_sequence):
            if target != predicted:
                # print(target, predicted)
                errors.append((target, predicted))
        return errors, Counter({'error_words': len(errors), 'total_words': len(target_sequence)})


    def evaluate_line(self, annotated_line, **kwargs):
        if len(annotated_line.strip()) > 0:
            target_sequence = self.helper.get_target_sequence(annotated_line)
            input_sequence = self.helper.get_input_sequence(target_sequence)
            predicted_sequence = self.act_on_line(input_sequence)
            # print(len(annotated_line), len(target_sequence), len(input_sequence), len(predicted_sequence))
            errors, word_result =  self.get_word_errors(target_sequence, predicted_sequence)
            try:
                char_result = self.get_character_errors(target_sequence, predicted_sequence, input_sequence)
            except:
                print(annotated_line)
                raise Exception()
            return errors, word_result + char_result
        else:
            return [], Counter({'error_words': 0, 'total_words': 0})

    def evaluate_file(self, annotated_file):
        error_list = []
        result = Counter()

        num_lines = get_num_lines(annotated_file)
        with open(annotated_file, 'r', encoding='utf-8') as infile:
            for annotated_line in tqdm(infile, total=num_lines):
                line_errors, line_result = self.evaluate_line(annotated_line)
                error_list.append(line_errors)
                result += line_result

        error_file = annotated_file + '.errors'
        with open(error_file, 'w', encoding='utf-8') as error_file:
            for line_no, line_errors in enumerate(error_list):
                if len(line_errors) > 0:
                    print("Line No.: {}".format(line_no), end=', ', file=error_file)
                    for target, predicted in line_errors:
                        print("Target: {}; Predicted: {}".format(target, predicted), end=' ', file=error_file)
                    print("", file=error_file)

            print("Total number of word errors: {}".format(result['error_words']), file=error_file)
            print("Total number of words: {}".format(result['total_words']), file=error_file)
            print("Word Accuracy: {}".format(1 - result['error_words']/result['total_words']), file=error_file)
            print("Word Accuracy: {}".format(1 - result['error_words'] / result['total_words']))
            print("Total number of char errors: {}".format(result['error_chars']), file=error_file)
            print("Total number of chars: {}".format(result['total_chars']), file=error_file)
            print("Character Accuracy: {}".format(1 - result['error_chars'] / result['total_chars']), file=error_file)
        return error_list, result


class Segmenter(Actor):
    """This class implements the segmentation functionality. It takes in the name of the model to load,
    the feature extactor and a label extractor in its constructor. Additionally, if you also pass in
    an object of Standardizer to the constructor, it will standardize the text before segmenting it"""

    def __init__(self, model, feature_extractor, label_extractor, dataset_helper=None, standardizer=None):
        super().__init__(model, feature_extractor, label_extractor, dataset_helper)
        self.standardizer = standardizer

    def act_on_word(self, word_no, line, **kwargs):
        word_features = np.array(self.fe.word_to_features(word_no, line))
        labels = self.model.predict(word_features)
        return self.le.labels_to_word(line[word_no], labels)

    def act_on_line(self, line):
        if self.standardizer:
            line = self.standardizer.act_on_line(line)
        line = clean_arabic(line)
        if len(line) == 0:
            return line
        else:
            no_space_flags = self.helper.get_nonspace_flags(line)
            features = self.fe.sent_to_features(line, no_space_flags)
            labels = self.model.predict(features)
            # print(len(line), len(labels))
            segmented_line = self.le.labels_to_sent(line, labels)
            return segmented_line

    def _act_on_line(self, line):
        if self.standardizer:
            line = self.standardizer.act_on_line(line)
        return super().act_on_line(line)

    def get_character_errors(self, target_sequence, predicted_sequence, *args):
        # print(len(target_sequence))
        # print(len(predicted_sequence))
        target_labels = self.le.sent_to_labels(target_sequence)
        predicted_labels = self.le.sent_to_labels(predicted_sequence)
        assert len(target_labels) == len(predicted_labels)

        error_chars, total_chars = 0, 0
        for target_char, predicted_char in zip(target_labels, predicted_labels):
            # print(target_char, predicted_char)
            if target_char != predicted_char:
                error_chars += 1
            total_chars += 1
        # assert len(target_sequence) == len(predicted_sequence)
        return Counter({'error_chars': error_chars, 'total_chars': total_chars})


class Standardizer(Actor):
    """This class implements the standardization functionality. It takes in a model, feature extractor
    and a label extractor. After creating the object, the following two functions can be called:
    1) act_on_line() and (2) act_on_file()
    The input sentence is first substandardized and features are extracted from this sentence. We then
    loop through the input sentence and standardize the necessary characters: """

    def __init__(self, model, feature_extractor, label_extractor, dataset_helper=None):
        super().__init__(model, feature_extractor, label_extractor, dataset_helper)

    # def act_on_char(self, char_no, word_no, line, **kwargs):
    #     substandard_line = kwargs['substandard_line']
    #     if self.le.should_standardize(char_no, word_no, line):
    #         char_features = self.fe.char_to_features(char_no, word_no, substandard_line)
    #         char_features = np.array(char_features).reshape(1, len(self.fe.template))
    #         label = self.model.predict(char_features)[0]
    #         return self.le.label_to_char(label)
    #     else:
    #         return line[word_no][char_no]

    def act_on_line(self, line, **kwargs):
        clean_line = self.helper.get_target_sequence(line)
        substandard_line = self.helper.get_input_sequence(clean_line)
        assert len(clean_line) == len(substandard_line)
        flags = self.helper.get_standardizable_flags(substandard_line)
        line_features = self.fe.sent_to_features(substandard_line, flags)
        labels = self.model.predict(line_features)
        standard_line = self.le.labels_to_sent(substandard_line, labels, flags)
        return standard_line

    def _act_on_line(self, line):
        clean_line = self.helper.get_target_sequence(line)
        substandard_line = self.helper.get_input_sequence(clean_line)
        assert len(clean_line) == len(substandard_line)
        result = list(clean_line)
        substandard_line = substandard_line.split()
        char_pos, word_pos = -1, 0 # Used to index characters in `substandard_line`
        for char_no, char in enumerate(result): # Used to index characters in `result`
            if char == ' ':
                char_pos = -1
                word_pos += 1
            else:
                char_pos += 1

            if self.helper.should_standardize(char_no, result):
                char_features = self.fe.char_to_features(char_pos, word_pos, substandard_line)
                char_features = np.array(char_features).reshape(1, len(self.fe.template))
                label = self.model.predict(char_features)[0]
                result[char_no] = self.le.label_to_char(label)
                # result[char_no] = 'ا'
        return ''.join(result)

    def get_character_errors(self, target_sequence, predicted_sequence, *args):
        target_labels = target_sequence
        predicted_labels = predicted_sequence
        input_sequence = args[0]
        assert len(target_sequence) == len(predicted_sequence)
        error_chars, total_chars = 0, 0
        for char_no, (target_char, predicted_char) in enumerate(zip(target_sequence, predicted_sequence)):
            if self.helper.should_standardize(char_no, input_sequence):
                if target_char != predicted_char:
                    error_chars += 1
                total_chars += 1
        return Counter({'error_chars': error_chars, 'total_chars': total_chars})


if __name__ == "__main__":
    s = "مغاني الشعب طيبا في المغاني بمنزلة الربيع من الزمان"
    # sso = substandardize_line(s)
    # std_model = pickle.load(open('models/lightgbm_standardizer.mod', 'rb'))
    seg_model = pickle.load(open('models/segmenter_catboost_325.mod', 'rb'))
    # std_le = StandardizerLE()
    seg_le = SegmenterLE()
    fe = OldFeatureExtractor('t1')
    seg_helper = SegmenterDatasetHelper('binary_plus')
    # std_helper = StandardizerDatasetHelper('eight_class')
    # std = Standardizer(std_model, fe, std_le, std_helper)
    seg = Segmenter(seg_model, fe, seg_le, seg_helper)
    # print(s)
    # print(sso)
    # print(std.act_on_line(s))
    # print(seg.act_on_line(s))
    # seg.act_on_file('sample/seg.txt', 'sample/seg.seg')
    # std.act_on_file('sample/small_siyar.sso', 'sample/small_siyar.std')
    files = ['data/segmenter/dev1.txt', 'data/segmenter/test1.txt', 'data/segmenter/test2.txt']
    for f in files:
        error_list, result = seg.evaluate_file(f)
        print(result)
    # error_list, result = std.evaluate_file('data/standardizer/siyar_1000.txt')
    # print(result)
    # print(std.evaluate_line(s))
    # print("Number of errors: {}".format(len(error_list)))
    # for error in error_list:
    #     print("Target: {}; Predicted: {}".format(error[0], error[1]))
        