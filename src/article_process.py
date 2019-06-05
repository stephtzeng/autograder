from nltk import sent_tokenize, word_tokenize
import pandas as pd
import argparse
import random
from collections import defaultdict
import os
import subprocess
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import url as sa_url
import re
import kenlm
from settings import conn_args

db_connect_url = sa_url.URL(**conn_args)
engine = sa.create_engine(db_connect_url)

def df_from_query(query):
    return pd.read_sql(query, engine)


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

        self._train_val_test_split()

        self.grade_level_sentences = dict() # each grade level will have a list of tokenized sentences
        self.models = dict()

    def _train_val_test_split(self):
        try:
            self.metadata_split = pd.read_csv(self.path_to_data + '/articles_metadata_split.csv')
            self.metadata_split = self.metadata_split[self.metadata_split.is_original == False]
        except FileNotFoundError:
            train = self.metadata.sample(frac=0.7, replace=False, random_state=1)
            test = self.metadata.drop(train.index)
            val = test.sample(frac=0.5, replace=False, random_state=1)
            test = test.drop(val.index)

            self.metadata_split.loc[train.index, 'train_val_test'] = 'train'
            self.metadata_split.loc[val.index, 'train_val_test'] = 'val'
            self.metadata_split.loc[test.index, 'train_val_test'] = 'test'

            is_original_q = """
            SELECT ah.slug AS slug,
                   -- al.article_header_id AS article_header_id,
                   CAST(al.grade_level AS int) AS grade_level,
                   al.is_original AS is_original
            FROM public.article_levels al
            JOIN public.article_headers ah
                ON al.article_header_id = ah.article_header_id
            WHERE ah.slug in {}
            """.format(tuple(self.metadata.slug.unique()))
            is_original = df_from_query(is_original_q)

            self.metadata_split = pd.merge(self.metadata, is_original, on=['slug', 'grade_level'], how='left')
            self.metadata_split.to_csv(self.path_to_data + '/articles_metadata_split.csv')

            self.metadata_split = self.metadata_split[self.metadata_split.is_original == False]

    def get_article_text(self, filename):
        """
        Returns
        :return:
        """
        txt = open(self.path_to_data + '/articles/' + filename, 'r')
        return txt.read()

    def split_article(self, article_text):
        """
        Ingests raw article text (text string)
        :param path:
        :return: List of strings (individual sentences)
        """
        # first split the text string on new lines to recognize paragraphs as different sentences
        paragraphs = article_text.split('\n')
        paragraphs = [i for i in paragraphs if len(i) > 0]  # makes sure that we do not include blank spaces
        all_sentences = []

        clean = re.compile('<.*?>')  # for removing html tags

        for p in paragraphs:
            sentences_list = sent_tokenize(p)
            sentences_list = [i for i in sentences_list if len(i) > 0]  # prevents blank sentences from being added
            # print('found {} sentences'.format(len(sentences_list)))
            for sentence in sentences_list:
                #             sentence = sentence.rstrip(
                #                 '.!?;').lower()  # remove ending punctuation and put the sentence into lower case
                sentence = sentence.lower()
                # data pre-processing step to remove embedded http links
                if len(sentence) > 1:
                    words = sentence.split(' ')
                    words = [i for i in words if 'http' not in i]  # pull out words that are actually html tags
                    if len(words) > 1:  # ignore one word sentences
                        sentence = ' '.join(words)
                # data pre-processing step to remove and html tags
                sentence = re.sub(clean, '', sentence)  # remove any html tags that are present
                if len(sentence) > 1:
                    # tokens = nltk.re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", sentence)
                    all_sentences.append(' '.join(word_tokenize(sentence)))
        return all_sentences

    def build_data(self):
        self.grade_level_sentences['train'] = dict()
        self.grade_level_sentences['val'] = dict()
        self.grade_level_sentences['test'] = dict()

        for row in self.metadata_split.itertuples():
            if row.language == 'en':
                text = self.get_article_text(row.filename)
                sentences = self.split_article(text)
                # sentences = [' '.join(nltk.word_tokenize(sentence)).lower() for sentence in sentences]

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
        # for gl in [2, 3, 4, 5, 6, 7, 8, 9]:
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

        sample_perp_df.loc[:, 'min_perplexity'] = sample_perp_df.idxmin(axis=1, skipna=True)
        sample_perp_df.loc[:, 'max_perplexity'] = sample_perp_df.idxmax(axis=1, skipna=True)
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