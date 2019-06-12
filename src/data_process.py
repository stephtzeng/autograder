'''
Data processing script to produce train/val/test sets for the RNN

Author: Kiley Yeakel
'''


import numpy as np
import glob
import pandas as pd
from nltk.tokenize import sent_tokenize
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from tqdm import tqdm
import re
import json
np.random.seed(42)

def read_article(path):
    """
    Pulls the article text
    :param path:
    :return: Full text string with no processing
    """
    txt = open(path)
    return txt.read()

def split_article(article_text, grade_level, filename):
    """
    Ingests raw article text (text string)
    :param path:
    :return: List of strings (individual sentences)
    """
    #first split the text string on new lines to recognize paragraphs as different sentences
    paragraphs = article_text.split('\n')
    paragraphs = [i for i in paragraphs if len(i) > 0] #makes sure that we do not include blank spaces
    all_sentences = []
    all_grades = []
    all_filenames = []
    clean = re.compile('<.*?>') #for removing html tags

    for p in paragraphs:
        sentences_list = sent_tokenize(p)
        sentences_list = [i for i in sentences_list if len(i) > 0] #prevents blank sentences from being added
        # print('found {} sentences'.format(len(sentences_list)))
        for sentence in sentences_list:
            sentence = sentence.rstrip('.!?;').lower() #remove ending punctuation and put the sentence into lower case

            #data pre-processing step to remove embedded http links
            if len(sentence) > 1:
                words = sentence.split(' ')
                words = [i for i in words if 'http' not in i] #pull out words that are actually html tags
                if len(words) > 1: #ignore one word sentences
                    sentence = ' '.join(words)

            #data pre-processing step to remove and html tags
            sentence = re.sub(clean, '', sentence) #remove any html tags that are present
            if len(sentence) > 1:
                # tokens = nltk.re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", sentence)
                all_sentences.append(sentence)
                all_grades.append(str(grade_level))
                all_filenames.append(filename)
    return all_sentences, all_grades, all_filenames


def main():
    path = '/Users/yeakekl1/PycharmProjects/CS229_project/Data'
    article_filenames = glob.glob(os.path.join(path, 'articles/*'))
    # print(np.shape(article_filenames))  # provide the number of articles currently in the database

    # load in the metadata file which provides the information on the articles language, title, grade level, version, filename
    library = pd.read_csv(os.path.join(path, 'articles_metadata_split.csv'))

    base_path = path + '/articles'
    all_sentences = []
    all_grades = []
    all_files = []

    # print('Parsing {} articles...'.format(len(library)))
    with tqdm(total = len(library)) as pbar:
        for index, row in library.iterrows():
            pbar.update(1)
            grade_level = row['grade_level']
            version = row['version']
            filename = row['filename']
            language = row['language']
            original = row['is_original']
            # if english, process it, otherwise do nothing
            if (language == 'en'):
                article_text = read_article(os.path.join(base_path, filename))
                sentences, grades, filenames = split_article(article_text, grade_level, filename)
                all_sentences.extend(sentences)
                all_grades.extend(grades)
                all_files.extend(filenames)

    #for each of the grades print out how many samples we have
    all_grades = np.asarray(all_grades)
    all_sentences = np.asarray(all_sentences)
    all_files = np.asarray(all_files)
    grades = np.unique(all_grades)

    all_sentences_capped = []
    all_grades_capped = []
    all_files_capped = []
    for g in grades:
        ind = np.where(all_grades == g)[0]
        print('for grade {} we have {} samples'.format(g, len(ind)))
        #create a binary grade classification
        all_sentences_capped.extend(all_sentences[ind])
        all_files_capped.extend(all_files[ind])
        if int(g) >= 6:
            label = 'Mid' #middle school level
        else:
            label = 'Elem' #elementary school
        labels = np.tile(label, (len(ind),1))
        all_grades_capped.extend(labels)

    # return all_sentences, all_grades, all_files, library
    # tokenize the sentences using the keras text processing function
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_sentences_capped)
    sequences = tokenizer.texts_to_sequences(all_sentences_capped)
    word_index = tokenizer.word_index
    print('Found {} unique words within the {} unique sentences'.format(len(word_index), len(all_sentences_capped)))

    # pad the sequences based on the length of the longest sentence
    len_sentence = [len(i) for i in sequences]
    MAXLEN = np.max(len_sentence)
    print('Maximum sentence length is {} words'.format(MAXLEN))
    X = pad_sequences(sequences, maxlen=MAXLEN)

    # clean up the labels and turn into a one-hot encoding
    all_grades_capped = np.asarray(all_grades_capped)
    classes = np.unique(all_grades_capped)
    class_assoc = {}
    for i,item in enumerate(classes):
        class_assoc[item] = i
    print(class_assoc)

    all_classes = np.zeros(np.shape(all_grades_capped))
    for i,item in enumerate(classes):
        #get key value
        key_value = class_assoc[item]
        ind_class = np.where(all_grades_capped == item)[0]
        all_classes[ind_class] = key_value
    #if we have more than two classes switch to one-hot encoding otherwise do binary encoding
    if len(classes) > 2:
        y = keras.utils.to_categorical(all_classes)
    else:
        y = np.asarray(all_classes)

    # split the data into train, validation and test sets
    # train/val/test = 80/10/10
    n = len(y)
    indices = np.arange(n)
    # shuffle the indices
    np.random.shuffle(indices)
    train_ind = indices[0 : int(0.8 * n)]
    val_ind = indices[int(0.8 * n) : int(0.9 * n)]
    test_ind = indices[int(0.9 * n) : n]

    # split the original datasets into train, validation and test sets
    X_train = X[train_ind, :]
    print('The shape of the X training set = {}'.format(np.shape(X_train)))
    X_val = X[val_ind, :]
    print('The shape of the X validation set = {}'.format(np.shape(X_val)))
    X_test = X[test_ind, :]
    print('The shape of the X test set = {}'.format(np.shape(X_test)))

    y_train = y[train_ind, :]
    print('The shape of the y training set = {}'.format(np.shape(y_train)))
    y_val = y[val_ind, :]
    print('The shape of the y validation set = {}'.format(np.shape(y_val)))
    y_test = y[test_ind, :]
    print('The shape of the y test set = {}'.format(np.shape(y_test)))

    #save the output to local directory
    np.savez('training_data_2class.npz', x = X_train, y = y_train)
    np.savez('validation_data_2class.npz', x = X_val, y = y_val)
    np.savez('test_data_2class.npz', x = X_test, y = y_test)

    with open('word_index_2class.json', 'w') as fp:
        json.dump(word_index, fp)
    with open('class_assoc_2class.json', 'w') as fp:
        json.dump(class_assoc, fp)

if __name__ == "__main__":
    main()


