import nltk
import pandas as pd
import argparse
import random
from collections import defaultdict
import os
import kenlm
import subprocess


class ArticleLM(object):
    """
    Article object stores metadata about each article
    """
    def __init__(self, path_to_data, path_to_kenlm, path_to_arpa):
        """

        :param path_to_data:
        :param path_to_arpa:
        :param path_to_kenlm:
        """
        self.path_to_data = path_to_data
        self.path_to_kenlm = path_to_kenlm
        self.path_to_arpa = path_to_arpa

        self.metadata = pd.read_csv(path_to_data + '/articles_metadata.csv')
        self.metadata.loc[:, 'grade_level'] = self.metadata.grade_level.astype('int')
        self.metadata_split = self.metadata.copy() # this stores all the splits in a new column

        self.train_val_test_split()

        self.grade_level_sentences = dict() # each grade level will have a list of tokenized sentences
        self.models = dict()

    def train_val_test_split(self):
        try:
            self.metadata_split = pd.read_csv(self.path_to_data + '/articles_metadata_split.csv')
        except FileNotFoundError:
            train = self.metadata.sample(frac=0.7, replace=False, random_state=1)
            test = self.metadata.drop(train.index)
            val = test.sample(frac=0.5, replace=False, random_state=1)
            test = test.drop(val.index)

            self.metadata_split.loc[train.index, 'train_val_test'] = 'train'
            self.metadata_split.loc[val.index, 'train_val_test'] = 'val'
            self.metadata_split.loc[test.index, 'train_val_test'] = 'test'

            self.metadata_split.to_csv(self.path_to_data + '/articles_metadata_split.csv')

    def get_article_text(self, filename):
        """
        Returns
        :return:
        """
        txt = open(self.path_to_data + '/articles/' + filename, 'r')
        return txt.read()

    def build_data(self):
        self.grade_level_sentences['train'] = dict()
        self.grade_level_sentences['val'] = dict()
        self.grade_level_sentences['test'] = dict()

        for row in self.metadata_split.itertuples():
            if row.language == 'en':
                text = self.get_article_text(row.filename)
                sentences = [' '.join(nltk.word_tokenize(sentence)).lower() for sentence in nltk.sent_tokenize(text)]

                if not self.grade_level_sentences[row.train_val_test].get(row.grade_level):
                    self.grade_level_sentences[row.train_val_test][row.grade_level] = sentences
                else:
                    self.grade_level_sentences[row.train_val_test][row.grade_level].extend(sentences)

    def text_sentences_for_grade(self, grade_level, split, verbose=False):

        # training text
        sentences = self.grade_level_sentences.get(split).get(grade_level)
        if verbose:
            print("Grade {} has {} sentences in the {} set.".format(grade_level,
                                                                    len(sentences),
                                                                    split))
        return '\n'.join(sentences)

    def train_arpa(self, grade_level):

        training_text = self.text_sentences_for_grade(grade_level, 'train', verbose=True)
        arpa_path = self.path_to_arpa + '/gl_%s.arpa' % grade_level
        lmplz_proc = subprocess.Popen([self.path_to_kenlm + '/build/bin/lmplz', '-o', '5', '--arpa', arpa_path],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = lmplz_proc.communicate(input=bytes(training_text, 'utf-8'))
        print(out[1].decode('utf-8'))

    def train_all_arpas(self):

        for grade_level in self.grade_level_sentences.get('train').keys():
            print('Processing {}'.format(grade_level))
            self.train_arpa(grade_level)

    def retrieve_models(self):
        for gl in self.grade_level_sentences['train'].keys():
            try:
                self.models[gl] = kenlm.LanguageModel(self.path_to_arpa + '/gl_{}.arpa'.format(gl))
            except OSError:
                print('Grade Level {} failed'.format(gl))

    def compute_sentence_perplexities(self, sentence):
        if not self.models:
            self.retrieve_models()

        # make the model key the index
        perplexity = []

        for gl in sorted(self.models.keys()):
            perplexity.append(self.models[gl].perplexity(sentence))
        return perplexity

    def compute_perplexity_validation(self, split_type, grade_level):
        # Split type can be 'val' or 'test'
        sample_perp = dict()
        for ix, sentence in enumerate(self.grade_level_sentences[split_type][grade_level]):
            perplexities = self.compute_sentence_perplexities(sentence)
            sample_perp[ix] = perplexities
        sample_perp_df = pd.DataFrame(sample_perp).T
        sample_perp_df.columns = sorted(self.models.keys())
        return sample_perp_df





# def print_grade_level_shapes(metadata_train, metadata_test, grade_level):
#     print("Train shape", metadata_train[metadata_train.grade_level == grade_level].shape)
#     print("Test shape", metadata_test[metadata_test.grade_level == grade_level].shape)



# def train_all_arpas(self):
#
#     grades = self.metadata.grade_level.unique().sort()
#     for grade_level in grades:
#         print('Processing {}'.format(grade_level))
#         training_text = self.text_sentences_for_grade(grade_level, verbose=True)
#         arpa_path = self.path_to_arpa + '/gl_%s.arpa' % grade_level
#         lmplz_proc = subprocess.Popen([self.path_to_kenlm, '-o', '5', '--arpa', arpa_path],
#                                       stdin=subprocess.PIPE,
#                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         lmplz_proc.communicate(training_text)

# def train_arpa(training_text, grade_level):
#     arpa_path = '/path-to-arpa-%s.arpa' % grade_level
#     lmplz_proc = subprocess.Popen(['/path/to/lmplz', '-o', '5', '--arpa', arpa_path],stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     lmplz_proc.communicate(training_text)




# for grade in grades:
#     train_arpa(sentences_for_grade(path, metadata_train, grade), grade)
#
#
# for grade in grades:
#     train_bert(sentences_for_grade(path, metadata_train, grade), grade)
#
#
# for grade in grades:
#     train_elmo(sentences_for_grade(path, metadata_train, grade), grade)