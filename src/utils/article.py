import pandas as pd
from collections import defaultdict


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
        self.library = None

    def create_library(self):
        self.library = defaultdict(Article)

        for row in self.metadata.itertuples():

            # do some kind of default dict thing where we add grade levels if article already exists
            if not self.library.get(row.slug):
                self.library[row.slug] = Article(row.slug, row.title, row.language, self.path)
            self.library[row.slug].add_grade_level_version_map(row.grade_level, row.version)


def main():
    # Showing how you could use this
    path = '/Users/stephanie/data/newsela_article_corpus_2016-01-29/'
    library = Library(path)
    library.create_library()
    print(library.library['zuckerberg-internet'].article_text(4.0))
    print(library.library['zuckerberg-internet'].grade_levels)



if __name__ == "__main__":
    main()
