import pandas as pd
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
import nltk


def get_article_text(path, slug, language, version):
    """
    Returns
    :return:
    """
    txt = open(path + 'articles/' + slug + '.' + language + '.'
               + str(version)
               + '.txt', 'r')
    return txt.read()


def tokenize_sentences(path, slug, language, grade_level):
    article_text = get_article_text(path, slug, language, grade_level)
    sentence_tokenize_list = sent_tokenize(article_text)
    all_words = []
    for sentence in sentence_tokenize_list:
        sentence = sentence.rstrip('.!?')
        tokens = nltk.re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", sentence)
        all_words.append('<s>')
        all_words.extend(tokens)
        all_words.append('</s>')

    return all_words


class Article(object):
    """
    Article object stores metadata about each article
    """
    def __init__(self, slug, title, language, path):
        """
        :param slug:
        :param title:
        :param language:
        :param path: same path as in library (see example in main)
        """
        self.slug = slug
        self.language = language
        self.title = title
        self.grade_level_to_version = dict()
        self.path = path

    def add_grade_level_version_map(self, grade_level, version):
        self.grade_level_to_version[grade_level] = version

    @property
    def grade_levels(self):
        """
        All available grade levels for this article
        :return: list of all grade levels
        """
        return self.grade_level_to_version.keys()

    def article_text(self, grade_level):
        """
        Returns
        :param grade_level:
        :return:
        """
        txt = open(self.path + 'articles/' + self.slug + '.' + self.language + '.'
                   + str(self.grade_level_to_version[grade_level])
                   + '.txt', 'r')
        return txt.read()

    def n_grams(self, grade_level, n):
        """
        Tokenizes by using space split and returns ngrams
        :param grade_level:
        :param n: n of n-gram (so 3 for a 3-gram)
        :return: ngrams object
        """
        tokens = [token for token in self.article_text(grade_level).split(" ") if token != ""]

        return ngrams(tokens, n)


class Library(object):
    """
    Library class holds all of the articles and metadata.
    We could rename this because library.library is a little confusing
    """
    def __init__(self, path):
        """
        :param path: the path that you're storing the text
        """
        self.path = path
        self.metadata = pd.read_csv(path + 'articles_metadata.csv')
        self.all_articles = defaultdict() # dict of dict of articles. May be unnecessary
        self.grade_level_tokenized_sentences = defaultdict()

        self.metadata.loc[:, 'grade_level'] = self.metadata.grade_level.astype('int')

    def create_grade_level_library(self):
        self.all_articles = defaultdict(Article)

        for row in self.metadata.itertuples():

            # do some kind of default dict thing where we add grade levels if article already exists
            if not self.all_articles.get(row.slug):
                self.all_articles[row.slug] = Article(row.slug, row.title, row.language, self.path)
            self.all_articles[row.slug].add_grade_level_version_map(row.grade_level, row.version)

            sent_tokens = tokenize_sentences(self.path,
                                    row.slug,
                                    row.language,
                                    row.version)

            if not self.grade_level_tokenized_sentences.get(row.grade_level):
                self.grade_level_tokenized_sentences[row.grade_level] = sent_tokens
            else:
                self.grade_level_tokenized_sentences[row.grade_level].extend(sent_tokens)

    def grade_level_vocabulary(self, grade_level):
        return set(self.grade_level_tokenized_sentences[grade_level])

    def grade_level_ngrams(self, grade_level, n):
        return list(ngrams(self.grade_level_tokenized_sentences.get(grade_level), n))


def main():
    # Showing how you could use this
    path = '/Users/stzeng/code/github/autograder/data/newsela_article_corpus_2016-01-29/'
    library = Library(path)
    library.create_grade_level_library()
    print(library.all_articles['zuckerberg-internet'].article_text(4.0))
    print(library.all_articles['zuckerberg-internet'].grade_levels)


if __name__ == "__main__":
    main()
