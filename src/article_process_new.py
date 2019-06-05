from nltk import sent_tokenize, word_tokenize
import pandas as pd
import argparse
import random
from collections import defaultdict
import os
import numpy as np
import subprocess
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import url as sa_url
import re
import kenlm
import matplotlib.pyplot as plt
import seaborn as sns
from settings import conn_args

db_connect_url = sa_url.URL(**conn_args)
engine = sa.create_engine(db_connect_url)

def df_from_query(query):
    return pd.read_sql(query, engine)


class ArticleLM(object):
    """
    Article object stores metadata about each article
    """
    def __init__(self, path_to_data, path_to_kenlm, path_to_arpa, n, level):
        """
        :param path_to_data:
        
        :param path_to_kenlm:
        :param path_to_arpa:
        :param n: n for n-gram as top-level (like 5 for 5-gram)
        :param level: can be "binary" or "level"
        
        """
        self.path_to_data = path_to_data
        self.path_to_kenlm = path_to_kenlm
        self.path_to_arpa = path_to_arpa

        self.metadata = pd.read_csv(path_to_data + '/MANIFEST.csv')
        self.metadata.loc[:, 'grade_level'] = self.metadata.grade_level.astype('int')
        self.metadata_split = self.metadata.copy() # this stores all the splits in a new column

        self.n = n
        self.level = level
        
        if self.level == 'binary':
            self.model_type = 'level'
        elif self.level == 'grade_level':
            self.model_type = 'grade_level'
        else:
            print("invalid input")
        

        self._train_val_test_split()

        self.level_sentences = dict() # each grade level points to a dict denoting train/test/val and
                                            #  each key in those dicts point to a list of tokenized sentences for the GL
                                            # will have a list of tokenized sentences

        ## for validation purposes
        self.article_val_sentences = dict() # each slug-level tuple  will have a list of tokenized sentences
        self.article_test_sentences = dict()

        self.models = dict()

    def _train_val_test_split(self):
        try:
            self.metadata_split = pd.read_csv(self.path_to_data + '/MANIFEST_split.csv')
            # self.metadata_split.loc[:, 'level'] = ['easy' if x <= 5 else 'hard' for x in self.metadata_split.grade_level]
            # self.metadata_split = self.metadata_split[(self.metadata_split.is_original == False)
            #                                           & (self.metadata_split.type == 'news')
            #                                           & (self.metadata_split.grade_level != 10)]

        except FileNotFoundError:
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

            self.metadata_split = pd.merge(self.metadata_split, is_original, on=['slug', 'grade_level'], how='left')
            self.metadata_split.loc[:, 'level'] = ['easy' if x <= 5 else 'hard' for x in self.metadata_split.grade_level]

            self.metadata_split = self.metadata_split[(self.metadata_split.is_original == False)
                                                      & (self.metadata_split.type == 'news')
                                                      & (self.metadata_split.grade_level != 10)]

            train = self.metadata_split.sample(frac=0.8, replace=False, random_state=1)
            test = self.metadata_split.drop(train.index)
            val = test.sample(frac=0.5, replace=False, random_state=1)
            test = test.drop(val.index)

            self.metadata_split.loc[train.index, 'train_val_test'] = 'train'
            self.metadata_split.loc[val.index, 'train_val_test'] = 'val'
            self.metadata_split.loc[test.index, 'train_val_test'] = 'test'

            self.metadata_split.to_csv(self.path_to_data + '/MANIFEST_split.csv')

    def get_article_text(self, file_path):
        """
        Returns
        :return:
        """
        txt = open(self.path_to_data + "/" + file_path, 'r')
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
        self.level_sentences['train'] = dict()
        self.level_sentences['val'] = dict()
        self.level_sentences['test'] = dict()

        self.article_val_sentences = dict()
        self.article_train_sentences = dict()

        for row in self.metadata_split.itertuples():
            if row.language == 'en':
                text = self.get_article_text(row.file_path)
                sentences = self.split_article(text)
                # sentences = [' '.join(word_tokenize(sentence)).lower() for sentence in sentences]

                if self.level == 'binary':
                    key = row.level
                elif self.level == 'grade_level':
                    key = row.grade_level

                if not self.level_sentences[row.train_val_test].get(key):
                    self.level_sentences[row.train_val_test][key] = sentences
                else:
                    self.level_sentences[row.train_val_test][key].extend(sentences)

                if row.train_val_test == 'val':
                    self.article_val_sentences[(row.slug, key)] = sentences
                if row.train_val_test == 'train':
                    self.article_train_sentences[(row.slug, key)] = sentences


    def text_sentences_for_grade(self, level, split, verbose=False):

        # training text
        sentences = self.level_sentences.get(split).get(level)
        if verbose:
            print("Grade {} has {} sentences in the {} set.".format(level,
                                                                    len(sentences),
                                                                    split))
        return '\n'.join(sentences)

    def train_arpa(self, level):

        training_text = self.text_sentences_for_grade(level, 'train', verbose=True)
        arpa_path = self.path_to_arpa + '/gl_%s_n%s.arpa' % (level, self.n)
        lmplz_proc = subprocess.Popen([self.path_to_kenlm + '/build/bin/lmplz', '-o', str(self.n), '--arpa', arpa_path],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = lmplz_proc.communicate(input=bytes(training_text, 'utf-8'))
        print(out[1].decode('utf-8'))

    def train_all_arpas(self):

        for level in self.level_sentences.get('train').keys():
            print('Processing {}'.format(level))
            self.train_arpa(level)

    def retrieve_models(self):
        for gl in self.level_sentences['train'].keys():
        # for gl in [2, 3, 4, 5, 6, 7, 8, 9]:
            try:
                self.models[gl] = kenlm.LanguageModel(self.path_to_arpa + '/gl_{}_n{}.arpa'.format(gl, self.n))
            except OSError:
                print('Grade Level {} failed'.format(gl))

    def compute_sentence_perplexities(self, sentence):
        """
        Computes the perplexity of each sentence and compares against all models.
        :param sentence:
        :return: list of perplexities against all models in models.keys(). Is indexed by sorted(self.models.keys())
        """
        if not self.models:
            self.retrieve_models()

        # make the model key the index
        perplexity = []

        for gl in sorted(self.models.keys()):
            perplexity.append(self.models[gl].perplexity(sentence))
        return perplexity

    def compute_perplexity_validation(self, split_type, level):
        """
        Each row is a different sentence
        :param split_type:
        :param level:
        :return:
        """
        # Split type can be 'val' or 'test'
        sample_perp = dict()
        for ix, sentence in enumerate(self.level_sentences[split_type][level]):
            perplexities = self.compute_sentence_perplexities(sentence)
            sample_perp[ix] = perplexities
        sample_perp_df = pd.DataFrame(sample_perp).T
        sample_perp_df.columns = sorted(self.models.keys())

        sample_perp_df.loc[:, 'min_perplexity'] = sample_perp_df.idxmin(axis=1, skipna=True)
        # sample_perp_df.loc[:, 'max_perplexity'] = sample_perp_df.idxmax(axis=1, skipna=True)
        return sample_perp_df

    def compute_all_sentences_best_guess(self, levels_considered):
        """
        levels_considered: [2, 3, 4, 5, 6, 7, 8, 9] as an example <-- unsure if this makes sense
        :param levels_considered:
        :return:
        """
        GL = dict()

        for gl in sorted(self.level_sentences['val'].keys()):
            GL[gl] = self.compute_perplexity_validation('val', gl)

        perplex_best_guess = []
        perplex_gl = []

        for gl in sorted(self.level_sentences['val'].keys()):
            perplex_best_guess.extend(GL[gl].min_perplexity)
            perplex_gl.extend([gl] * len(GL[gl].min_perplexity))

        perplex_guesses = pd.DataFrame({'level': perplex_gl, 'best_guess': perplex_best_guess})
        return perplex_guesses

    def plot_sentence_distribution(self, levels_considered):
        """
        Repetitive with `compute_all_sentences_best_guess()` so may move out
        Can only plot for grade levels, not binary
        :param levels_considered:
        :return:
        """
        perplex_guesses = self.compute_all_sentences_best_guess(levels_considered)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca()
        ax.set_title("Grade Level Distributions")
        # ax.set_xlabel("True Grade Level")
        # ax.set_ylabel("Predicted Grade Level")
        sns.boxplot(ax=ax, x="level", y="best_guess", data=perplex_guesses)
        ax.set_xlabel("True Grade Level")
        ax.set_ylabel("Predicted Grade Level")

        return perplex_guesses

    def compute_perplexity_articles(self, split_type):
        """
        Compute the best model for each article by looking at mean, median, or mode
          model with min perplexity
        :param split_type:
        :return:
        """
        if split_type == 'val':
            # article sentences here is a dict, key is article, maps to all its sentences
            article_sentences = self.article_val_sentences
        elif split_type == 'test':
            article_sentences = self.article_test_sentences

        iter = 0
        article_df = pd.DataFrame({})
        for article_gl in article_sentences.keys():
            #     mean_guess, median_guess = self.compute_perplexity_entire_article(article_sentences[article_gl])

            sample_perp = dict()
            for ix, sentence in enumerate(article_sentences[article_gl]):
                perplexities = self.compute_sentence_perplexities(sentence)
                sample_perp[ix] = perplexities
            sample_perp_df = pd.DataFrame(sample_perp).T
            sample_perp_df.columns = sorted(self.models.keys())

            sample_perp_df.loc[:, 'min_perplexity'] = sample_perp_df.idxmin(axis=1, skipna=True)
            # sample_perp_df.loc[:, 'max_perplexity'] = sample_perp_df.idxmax(axis=1, skipna=True)
            sample_perp_df.loc[:, 'true_gl'] = article_gl[1]
            sample_perp_df.loc[:, 'article'] = article_gl[0]

            article_df = pd.concat([article_df, sample_perp_df])

            iter += 1

            if iter % 100 == 0:
                print("iteration {}, article {}".format(iter, article_gl))

        return article_df

    def compute_article_best_grade_levels(self):
        article_sentences = self.article_val_sentences

        iter = 0
        article_df = pd.DataFrame({})
        for article_gl in article_sentences.keys():
            #     mean_guess, median_guess = self.compute_perplexity_entire_article(article_sentences[article_gl])

            sample_perp = dict()
            for ix, sentence in enumerate(article_sentences[article_gl]):
                perplexities = self.compute_sentence_perplexities(sentence)
                sample_perp[ix] = perplexities
            sample_perp_df = pd.DataFrame(sample_perp).T
            sample_perp_df.columns = sorted(self.models.keys())

            sample_perp_df.loc[:, 'min_perplexity'] = sample_perp_df.idxmin(axis=1, skipna=True)
            sample_perp_df.loc[:, 'true_gl'] = article_gl[1]
            sample_perp_df.loc[:, 'article'] = article_gl[0]

            article_df = pd.concat([article_df, sample_perp_df])

            iter += 1

            if iter % 10 == 0:
                print("iteration {}, article {}".format(iter, article_gl))

        means = article_df.groupby(['article', 'true_gl']).mean()['min_perplexity'].reset_index()
        means.rename(columns={'min_perplexity': 'predicted_gl'}, inplace=True)

        return means

    def plot_article_best_grade_levels(self, means):
        """

        :param means: the output of function `compute_article_best_grade_levels`
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca()
        ax.set_title("Grade Level Distributions")
        # ax.set_xlabel("True Grade Level")
        # ax.set_ylabel("Predicted Grade Level")
        sns.boxplot(ax=ax, x="true_gl", y="predicted_gl", data=means)
        ax.set_xlabel("True Grade Level")
        ax.set_ylabel("Predicted Grade Level, using Means")

    def compute_article_best_guess_binary(self):
        """
        For binary models, returns df containing best guess for the entire article - hard or easy?
        :return:
        """
        article_sentences = self.article_val_sentences

        iter = 0
        article_df = pd.DataFrame({})
        for article_gl in article_sentences.keys():
            #     mean_guess, median_guess = self.compute_perplexity_entire_article(article_sentences[article_gl])

            sample_perp = dict()
            for ix, sentence in enumerate(article_sentences[article_gl]):
                perplexities = self.compute_sentence_perplexities(sentence)
                sample_perp[ix] = perplexities
            sample_perp_df = pd.DataFrame(sample_perp).T
            sample_perp_df.columns = sorted(self.models.keys())

            sample_perp_df.loc[:, 'min_perplexity'] = sample_perp_df.idxmin(axis=1, skipna=True)
            sample_perp_df.loc[:, 'true_gl'] = article_gl[1]
            sample_perp_df.loc[:, 'article'] = article_gl[0]

            article_df = pd.concat([article_df, sample_perp_df])

            iter += 1

            if iter % 10 == 0:
                print("iteration {}, article {}".format(iter, article_gl))

        most_common = article_df.groupby(['article', 'true_gl']).agg(lambda x: x.value_counts().index[0])
        most_common.rename(columns={'min_perplexity': 'predicted_gl'}, inplace=True)
        most_common = most_common.reset_index()

        return most_common

    #
    # def compute_article_best_guess(self, split_type):
    #     """
    #
    #     :return:
    #     """
    #     if split_type == 'val':
    #         # article sentences here is a dict, key is article, maps to all its sentences
    #         article_sentences = self.article_val_sentences
    #     elif split_type == 'test':
    #         article_sentences = self.article_test_sentences
    #
    #     best_guess_mean = []  # note that these are going to be _indices_ not actual GL yet
    #     best_guess_median = []
    #     true_gl = []
    #
    #     for article_gl in article_sentences.keys():
    #         mean_guess, median_guess = self.compute_perplexity_entire_article(article_sentences[article_gl])
    #         best_guess_mean.append(mean_guess)
    #         best_guess_median.append(median_guess)
    #         true_gl.append(article_gl[1])
    #
    #     return pd.DataFrame({'true_level': true_gl, 'mean_best_guess': best_guess_mean,
    #                          'median_best_guess': best_guess_median})


