"""
article_lm will allow you to train language models using the KenLM package and find best classes, both
using multi-class classification as well as binary.

author: Stephanie Tzeng
"""

from nltk import sent_tokenize, word_tokenize
import pandas as pd
import random
from collections import defaultdict
import os
import numpy as np
import subprocess
import pandas as pd
import re
import kenlm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression


class ArticleLM(object):
    """
    Article object stores metadata about each article
    """

    def __init__(self, path_to_data, path_to_kenlm, path_to_arpa, n, level, keep_orig=False):
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

        self.metadata = pd.read_csv(path_to_data + '/articles_metadata.csv')
        self.metadata.loc[:, 'grade_level'] = self.metadata.grade_level.astype('int')
        self.metadata_split = self.metadata.copy()  # this stores all the splits in a new column
        self.keep_orig = keep_orig

        self.n = n
        self.level = level

        if self.level == 'binary':
            self.model_type = 'level'
        elif self.level == 'grade_level':
            self.model_type = 'grade_level'
        else:
            print("invalid input")

        self._train_val_test_split()

        self.level_sentences = dict()  # each grade level points to a dict denoting train/test/val and
        #  each key in those dicts point to a list of tokenized sentences for the GL
        # will have a list of tokenized sentences

        ## for validation purposes
        self.article_val_sentences = dict()  # each slug-level tuple  will have a list of tokenized sentences
        self.article_test_sentences = dict()

        self.models = dict()

    def _train_val_test_split(self):
        try:
            self.metadata_split = pd.read_csv(self.path_to_data + '/articles_metadata_split.csv')
            if not self.keep_orig:
                self.metadata_split = self.metadata_split[self.metadata_split.is_original == False]

        except FileNotFoundError:
            self.metadata_split.loc[:, 'level'] = ['easy' if x <= 5 else 'hard' for x in
                                                   self.metadata_split.grade_level]

            self.metadata_split = self.metadata_split[
                # (self.metadata_split.type == 'news') &
                (self.metadata_split.grade_level != 10)]

            train = self.metadata_split.sample(frac=0.8, replace=False, random_state=1)
            test = self.metadata_split.drop(train.index)
            val = test.sample(frac=0.5, replace=False, random_state=1)
            test = test.drop(val.index)

            self.metadata_split.loc[train.index, 'train_val_test'] = 'train'
            self.metadata_split.loc[val.index, 'train_val_test'] = 'val'
            self.metadata_split.loc[test.index, 'train_val_test'] = 'test'

            self.metadata_split.to_csv(self.path_to_data + '/articles_metadata_split.csv')

            if not self.keep_orig:
                self.metadata_split = self.metadata_split[self.metadata_split.is_original == False]

    def get_article_text(self, filename):
        """
        :return: text from article
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
            for sentence in sentences_list:
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
                    all_sentences.append(' '.join(word_tokenize(sentence)))
        return all_sentences

    def build_data(self):
        """
        Build all of the data for the model. Must be executed pull in data
        """
        self.level_sentences['train'] = dict()
        self.level_sentences['val'] = dict()
        self.level_sentences['test'] = dict()

        self.article_val_sentences = dict()
        self.article_train_sentences = dict()

        for row in self.metadata_split.itertuples():
            if row.language == 'en':
                text = self.get_article_text(row.filename)
                sentences = self.split_article(text)

                if self.level == 'binary':
                    key = row.level
                elif self.level == 'grade_level':
                    key = row.grade_level

                if not self.level_sentences[row.train_val_test].get(key):
                    self.level_sentences[row.train_val_test][key] = sentences
                else:
                    self.level_sentences[row.train_val_test][key].extend(sentences)

                if row.train_val_test == 'train':
                    self.article_train_sentences[(row.slug, key)] = sentences
                if row.train_val_test == 'val':
                    self.article_val_sentences[(row.slug, key)] = sentences
                if row.train_val_test == 'test':
                    self.article_test_sentences[(row.slug, key)] = sentences

    def text_sentences_for_grade(self, level, split, verbose=False):

        # training text
        sentences = self.level_sentences.get(split).get(level)
        if verbose:
            print("Grade {} has {} sentences in the {} set.".format(level,
                                                                    len(sentences),
                                                                    split))
        return '\n'.join(sentences)

    def train_arpa(self, level):
        """
        Trains a single arpa file depending on the level chosen.
        :param level: for self.level == 'binary', can be either 'easy' or 'hard'
                      for self.level == 'grade_level', is the grade level in question
        """

        training_text = self.text_sentences_for_grade(level, 'train', verbose=True)
        arpa_path = self.path_to_arpa + '/gl_%s_n%s.arpa' % (level, self.n)
        lmplz_proc = subprocess.Popen([self.path_to_kenlm + '/build/bin/lmplz', '-o', str(self.n), '--arpa', arpa_path],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = lmplz_proc.communicate(input=bytes(training_text, 'utf-8'))
        print(out[1].decode('utf-8'))

    def train_all_arpas(self):
        """
        Trains all arpas at once
        """

        for level in self.level_sentences.get('train').keys():
            print('Processing {}'.format(level))
            self.train_arpa(level)

    def retrieve_models(self):
        for gl in self.level_sentences['train'].keys():
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
        elif split_type  == 'train':
            article_sentences = self.article_train_sentences

        if not self.models:
            self.retrieve_models()

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

            if iter % 100 == 0:
                print("iteration {}, article {}".format(iter, article_gl))

        article_df.rename(columns={'min_perplexity': 'predicted_gl'}, inplace=True)

        return article_df

    def _compute_article_best_guess_grade_levels(self, article_df):

        means = article_df.groupby(['article', 'true_gl']).mean()['predicted_gl'].reset_index()

        return means

    def plot_article_best_grade_levels(self, means):
        """

        :param means: the output of function `compute_article_best_grade_levels`
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca()
        ax.set_title("Grade Level Distributions")
        sns.boxplot(ax=ax, x="true_gl", y="predicted_gl", data=means)
        ax.set_xlabel("True Grade Level")
        ax.set_ylabel("Predicted Grade Level, using Means")

    def _compute_article_best_guess_binary(self, article_df):
        """
        For binary models, returns df containing best guess for the entire article - hard or easy?
        :return:
        """

        most_common = article_df.groupby(['article', 'true_gl']).agg(lambda x: x.value_counts().index[0])
        most_common = most_common.reset_index()

        return most_common


    def compute_article_to_sentence_guess(self, split_type):
        """

        :param split_type: str, input can be "train", "test", or "val"
        :return article_df: dataframe containing a row for each estimated complexity level
                            for each sentence for each article:true grade level combination
        """

        if split_type == 'val':
            # article sentences here is a dict, key is article, maps to all its sentences
            article_sentences = self.article_val_sentences
        elif split_type == 'test':
            article_sentences = self.article_test_sentences
        elif split_type == 'train':
            article_sentences = self.article_train_sentences

        iter = 0
        article_df = pd.DataFrame({})
        for article_gl in article_sentences.keys():

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

            if iter % 100 == 0:
                print("iteration {}, article {}".format(iter, article_gl))

        article_df.rename(columns={'min_perplexity': 'predicted_gl'}, inplace=True)

        return article_df

    def compute_article_best_guess(self, split_type):
        """
        Computes the best guess of the entire article depending on method
        Possible ways:
            - Binary classification (easy vs hard)
            - Using means on each grade level
        :return:
            article_df: dataframe containing a row for each estimated complexity level
                            for each sentence for each article:true grade level combination
            best_guess: dataframe containing each article:true grade level and "best guess"
        """

        article_df = self.compute_article_to_sentence_guess(split_type)

        if self.level == 'binary':
            best_guess = self._compute_article_best_guess_binary(article_df)
            return article_df, best_guess
        elif self.level == 'grade_level':
            best_guess = self._compute_article_best_guess_grade_levels(article_df)
            return article_df, best_guess

    def fit_linear_regression(self, article_train_df):
        """
        Fits linear regression from the training data to predict grade level of the entire article
        It will take all articles, calculate % of each predicted grade level for each sentence,
        and use those as inputs to predict a grade level (y)
        :param article_train_df: output of `compute_article_to_sentence_guess('train')`
        :return: clf (the fitted linear regression)
        """

        article_pred_gl_count = (np.array(article_train_df[['true_gl', 'article', 'predicted_gl']]
                                          .pivot_table(columns='predicted_gl',
                                                       index=['true_gl', 'article'],
                                                       aggfunc=lambda x: len(x),
                                                       fill_value=0)))

        y = (np.array(article_train_df[['true_gl', 'article', 'predicted_gl']]
                      .pivot_table(columns='predicted_gl',
                                   index=['true_gl', 'article'],
                                   aggfunc=lambda x: len(x),
                                   fill_value=0).reset_index().true_gl))

        X = np.divide(article_pred_gl_count, article_pred_gl_count.sum(axis=1).reshape(len(article_pred_gl_count), 1))

        clf = LinearRegression()
        clf.fit(X, y)

        return clf

    def compute_article_linear_regression_best_guess(self, article_split_df, clf):
        """

        :param article_split_df: output of `compute_article_to_sentence_guess()` depending on split_type
        :param clf: output of fit_linear_regression
        :return: error, accuracy, f1 score
        """
        split_df = (article_split_df[['true_gl', 'article', 'predicted_gl']]
                    .pivot_table(columns='predicted_gl',
                                 index=['true_gl', 'article'],
                                 aggfunc=lambda x: len(x),
                                 fill_value=0).reset_index()[['true_gl', 'article']])

        split_df.loc[:, 'predicted_gl'] = list(clf.predict(X))

        split_df.loc[:, 'predicted_gl'] = split_df.predicted_gl.astype('int')

        split_df.loc[:, 'true_easy'] = [1 if x <= 5 else 0 for x in split_df.true_gl]
        split_df.loc[:, 'predicted_easy'] = [1 if x <= 5 else 0 for x in split_df.predicted_gl]

        error = 1 - accuracy_score(split_df.true_easy, split_df.predicted_easy)
        accuracy = accuracy_score(split_df.true_easy, split_df.predicted_easy)
        f1 = f1_score(split_df.true_easy, split_df.predicted_easy)

        return error, accuracy, f1


    def compute_scores_grade_level(self, best_guess):
        """
        Scores, only for grade-level models. Not valid for binary model.
        :param best_guess: output of `compute_article_best_guess`
        :return: grade_level_cm: confusion matrix for grade levels
                 raw_accuracy: accuracy for predicted grade levels
                 banded_accuracy: accuracy for +/- grade level predictions
                 binary_cm: binary confusion matrix
                 binary_accuracy: accuracy for easy/hard using grade level models
                 binary_f1: f1 score for easy/hard using grade level models
        """
        if self.level == "binary":
            print("Wrong function!")
            return None, None, None, None, None, None

        best_guess.loc[:, 'predicted_gl'] = best_guess.predicted_gl.astype('int')
        best_guess.loc[:, 'true_easy'] = [1 if x <= 5 else 0 for x in best_guess.true_gl]
        best_guess.loc[:, 'predicted_easy'] = [1 if x <= 5 else 0 for x in best_guess.predicted_gl]

        # Creating a 'within 1 grade' band
        best_guess.loc[:, 'within_1_grade'] = list(map(lambda x, y: 1
                                                       if (y >= x - 1 and y <= x + 1)
                                                       else 0, best_guess.true_gl, best_guess.predicted_gl))

        grade_level_cm = confusion_matrix(best_guess.true_gl, best_guess.predicted_gl)
        raw_accuracy = accuracy_score(best_guess.true_gl, best_guess.predicted_gl)
        banded_accuracy = len(best_guess[best_guess.within_1_grade == 1]) * 1. / len(best_guess)

        binary_cm = confusion_matrix(best_guess.true_easy, best_guess.predicted_easy)
        binary_accuracy = accuracy_score(best_guess.true_easy, best_guess.predicted_easy)

        binary_f1 = f1_score(best_guess.true_easy, best_guess.predicted_easy, pos_label=0)

        return (grade_level_cm, raw_accuracy, banded_accuracy,
                binary_cm, binary_accuracy, binary_f1)


    def compute_scores_binary(self, best_guess):
        """
        Scores, only for binary models. Not valid for grade_level model.
        :param best_guess: output of `compute_article_best_guess`
        :return: cm: confusion matrix for easy/hard
                 binary_accuracy: accuracy for easy/hard using binary models
                 binary_f1: f1 score for easy/hard using binary models
        """
        if self.level == "grade_level":
            print("Wrong function!")
            return (None, None, None)

        cm = confusion_matrix(best_guess.true_gl, best_guess.predicted_gl)

        binary_accuracy = accuracy_score(best_guess.true_gl, best_guess.predicted_gl)

        binary_f1 = f1_score(best_guess.true_gl, best_guess.predicted_gl, pos_label='easy')

        return cm, binary_accuracy, binary_f1
