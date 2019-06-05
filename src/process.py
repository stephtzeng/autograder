import nltk
import pandas as pd
import argparse
import random
from collections import defaultdict
import os
import kenlm

# http://victor.chahuneau.fr/notes/2012/07/03/kenlm.html
# Run: python process.py | ~/github/kenlm/build/bin/lmplz -o 3 > eighth.arpa

def get_article_text(path, filename):
    """
    Returns
    :return:
    """
    txt = open(path + 'articles/' + filename, 'r')
    return txt.read()

def print_sentences_for_arpa_processing(path, metadata, grade_level):
    for row in metadata.itertuples():

        if row.grade_level == grade_level:
            text = get_article_text(path, row.filename)

            for sentence in nltk.sent_tokenize(text):
                print(' '.join(nltk.word_tokenize(sentence)).lower())

def print_grade_level_shapes(metadata_train, metadata_test, grade_level):
    print("Train shape", metadata_train[metadata_train.grade_level == grade_level].shape)
    print("Test shape", metadata_test[metadata_test.grade_level == grade_level].shape)

## Move this to another file for preprocessing
# metadata_train = metadata.sample(frac=0.7, replace=False, random_state=1)
# metadata_test = metadata.drop(metadata_train.index)
# metadata_train.to_csv(path + 'articles_metadata_train.csv')
# metadata_test.to_csv(path + 'articles_metadata_test.csv')

def main():
    # path = '/Users/stzeng/code/github/autograder/data/newsela_article_corpus_2016-01-29/'
    path = '/Users/stephanie/data/newsela_article_corpus_2016-01-29/'
    metadata = pd.read_csv(path + 'articles_metadata.csv')
    metadata.loc[:, 'grade_level'] = metadata.grade_level.astype('int')
    metadata_train = pd.read_csv(path + 'articles_metadata_train.csv')
    metadata_test = pd.read_csv(path + 'articles_metadata_test.csv')

    parser = argparse.ArgumentParser(description='Process grade_level')
    parser.add_argument('grade_level', type=int,
                        help='an integer for grade level')
    args = parser.parse_args()

    # print_grade_level_shapes(metadata_test, metadata_train, args.grade_level)
    print_sentences_for_arpa_processing(path, metadata_train, args.grade_level)


if __name__ == "__main__":
    main()