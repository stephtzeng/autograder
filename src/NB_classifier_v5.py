'''
This script will contain the data processing and model training for a Naive Bayes classifier for varying n_grams,
varying text lengths and varying numbers of classes

Includes secondary linear classifier

All model runs are saved to a dictionary and output as a pickle file

Author: Kiley Yeakel

'''

import numpy as np
import glob
import pandas as pd
from nltk.tokenize import sent_tokenize
import os
from tqdm import tqdm
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle
np.random.seed(42)

def read_article(path):
    """
    Pulls the article text
    :param path:
    :return: Full text string with no processing
    """
    txt = open(path)
    return txt.read()

def clean_article(article_text, num_sentences):
    """
    Ingests raw article text (text string)
    Cleans the article text to remove HTML tags, etc
    :param path:
    num_sentences = number of sentences to concatenate together for the word counting
    NB will be performed on the concatenated strings
    num_senetences = -1 implies the entire article (i.e., NB will be performed on the entire article)
    :return: List of strings depending on num_sentences and the total umber of sentences in the article
    """
    #first split the text string on new lines to recognize paragraphs as different sentences
    paragraphs = article_text.split('\n')
    paragraphs = [i for i in paragraphs if len(i) > 0] #makes sure that we do not include blank spaces
    all_sentences = []
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

    if num_sentences == -1:
        text = " ".join(all_sentences)
    else:
        text = []
        for i in range(len(all_sentences))[::num_sentences]:
            tmp = all_sentences[i:i+num_sentences]
            if(len(tmp) == num_sentences):
                tmp = " ".join(tmp)
                text.append(tmp)
    return text

#build up a dictionary of split up sentences

def create_naive_bayes_datasets(num_sentences, num_classes, include_orig):
    '''
    Reads in articles, processes the data to remove html links, splits the data according to the number of sentences
    :param num_sentences: length of text segments in sentences
    :param num_classes: identifies as binary or multiclass classification
    :param include_orig: flag idicating whether or not to include the original set of articles
    :return:
    list of strings correponding to segments of articles, labels for the text segments, article ID for each text segment
    '''

    path = '/Users/yeakekl1/PycharmProjects/CS229_project/Data'
    article_filenames = glob.glob(os.path.join(path, 'articles/*'))
    # print(np.shape(article_filenames))  # provide the number of articles currently in the database

    # load in the metadata file which provides the information on the articles language, title, grade level, version, filename
    library = pd.read_csv(os.path.join(path, 'articles_metadata_split.csv'))

    base_path = path + '/articles'
    all_articles = []
    all_grades = []
    all_files = []
    all_order = [] # tracks the index of an sentence grouping within a given article

    print('Parsing {} articles for sentence length = {}, classes = {}'.format(len(library), num_sentences, num_classes))
    with tqdm(total = len(library)) as pbar:
        for index, row in library.iterrows():
            pbar.update(1)
            grade_level = row['grade_level']
            version = row['version']
            filename = row['filename']
            language = row['language']
            original = row['is_original']
            # if we do not want to include the original articles
            if include_orig == False:
                if (language == 'en') and (original == False):
                    article_text = read_article(os.path.join(base_path, filename))
                    article_text = clean_article(article_text, num_sentences)
                    if num_sentences == -1: # include the entire article text
                        all_articles.append(article_text)
                        all_grades.append(grade_level)
                        all_files.append(filename)
                        all_order.append([0])
                    else: # extend to include all the subtexts of the article
                        all_articles.extend(article_text)
                        all_grades.extend(np.tile(grade_level, len(article_text)))
                        all_files.extend(np.tile(filename, len(article_text)))
                        all_order.extend(np.arange(len(article_text)))
            else:
                if (language == 'en'):
                    article_text = read_article(os.path.join(base_path, filename))
                    article_text = clean_article(article_text, num_sentences)
                    if num_sentences == -1: # include the entire article text
                        all_articles.append(article_text)
                        all_grades.append(grade_level)
                        all_files.append(filename)
                        all_order.append([0])
                    else: # extend to include all the subtexts of the article
                        all_articles.extend(article_text)
                        all_grades.extend(np.tile(grade_level, len(article_text)))
                        all_files.extend(np.tile(filename, len(article_text)))
                        all_order.extend(np.arange(len(article_text)))

    #for each of the grades print out how many samples we have
    all_grades = np.asarray(all_grades)
    all_articles = np.asarray(all_articles)
    all_files = np.asarray(all_files)
    grades = np.unique(all_grades)

    if num_classes == 2:
        y = np.zeros(np.shape(all_grades))
        y[all_grades >= 6] = 1
    elif num_classes == 5:
        grades_to_keep = [4, 5, 6, 7, 8]
        crit = [i in grades_to_keep for i in all_grades]
        ind_keep = np.where(np.asarray(crit) == True)[0]
        y = all_grades[ind_keep]
        all_articles = all_articles[ind_keep]
        all_files = all_files[ind_keep]
    elif num_classes == 9:
        grades_to_keep = [2, 3, 4, 5, 6, 7, 8, 9, 12]
        crit = [i in grades_to_keep for i in all_grades]
        ind_keep = np.where(np.asarray(crit) == True)[0]
        y = all_grades[ind_keep]
        all_articles = all_articles[ind_keep]
        all_files = all_files[ind_keep]
    print('Total number of text samples = {}'.format(len(all_articles)))

    # turn the all_files parameter into a unique article ID
    files_uni = np.unique(all_files)
    art_id = np.zeros(len(all_files))
    for f, item in enumerate(files_uni):
        art_id[all_files == item] = f

    return all_articles, y, art_id

def split_and_transform_data(X, y, art_id, n_grams):
    '''
    Splits data into training, validation and test sets based on article ID
    Produces count vectors of the data by calculating a dictionary used the n-gram length
    Performs TFIDF transform on the training data
    :param X: list of strings corresponding to text segments of given numbers of sentences
    :param y: labels associated with the strings
    :param art_id: article ID associated with the strings
    :param n_grams: max n-gram length to use in determining dictionary
    :return:
    Training, validation and test sets as well as dictionary and ignored words

    In computing count vectors will ignore n-grams that appear in less than 5 text segments or more than
    1000 text segments
    '''

    # split the data into train, validation and test sets
    # train/val/test = 80/10/10

    #split data based on entire article
    n = np.unique(art_id)
    # shuffle the indices
    np.random.shuffle(n)
    train_id = np.arange(0, int(0.8*len(n)))
    val_id = np.arange(int(0.8 * len(n)), int(0.9 * len(n)))
    test_id = np.arange(int(0.9 * len(n)), len(n))

    train_ind = []
    for t in train_id:
        tmp = np.where(art_id == t)[0]
        train_ind.extend(tmp)

    val_ind = []
    for v in val_id:
        tmp = np.where(art_id == v)[0]
        val_ind.extend(tmp)

    test_ind = []
    for s in test_id:
        tmp = np.where(art_id == s)[0]
        test_ind.extend(tmp)

    #transform all of the X data into counts
    vectorizer = CountVectorizer(ngram_range=n_grams, min_df = 5, max_df = 1000)
    X = vectorizer.fit_transform(X)  # returns the counts
    dictionary = vectorizer.vocabulary_
    words = vectorizer.get_feature_names()
    words = np.asarray(words)
    print('The length of the dictionary is: {}'.format(len(dictionary)))
    stop_words = vectorizer.stop_words_
    print('The number of words excluded from the dictionary is: {}'.format(len(stop_words)))

    # split the original datasets into train, validation and test sets
    X_train = X[train_ind]
    print('The shape of the X training set = {}'.format(np.shape(X_train)))
    X_val = X[val_ind]
    print('The shape of the X validation set = {}'.format(np.shape(X_val)))
    X_test = X[test_ind]
    print('The shape of the X test set = {}'.format(np.shape(X_test)))

    y_train = y[train_ind]
    print('The shape of the y training set = {}'.format(np.shape(y_train)))
    y_val = y[val_ind]
    print('The shape of the y validation set = {}'.format(np.shape(y_val)))
    y_test = y[test_ind]
    print('The shape of the y test set = {}'.format(np.shape(y_test)))


    #do a tfidf transformation on the training data only
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train)  # returns frequences

    # return of a dictionary of the classes
    classes = np.unique(y)
    print(classes)

    return X, X_train_tfidf, X_val, X_test, y_train, y_val, y_test, words, stop_words, classes, train_id, val_id, test_id

def train_NB_model(X_train, y_train):
    '''
    Trains a multinomial NB model
    :param X_train: training set
    :param y_train: labels
    :return: model
    '''
    print('Training the model...')
    nb = MultinomialNB()
    model = nb.fit(X_train, y_train)
    return model

def predict_from_NB_model(model, X_test):
    '''
    Calculates predictions of the NB model
    :param model: NB model
    :param X_test: test set
    :return:
    '''
    y_pred = model.predict(X_test)
    return y_pred

def create_article_dataset(pred, labels, text_id, train_id, val_id, test_id):
    '''
    This function will create a dataset for the secondary level in which predictions on individual sub-texts are accumulated into an article level prediction
    Split the predictions from the NB classifer on the article level using text_id
    :param pred: predictions for each text segment from the NB classifier
    :param labels: labels for each text segment
    :param text_id: identifes each text segment as belonging to a particular article
    :param train_id: article IDs used for the training set (consistent with NB model)
    :param val_id: article IDs used for the validation set (consistent with NB model)
    :param test_id: article IDs used for the test set (consistent with NB model)
    :return:
    creates two different datasets for use by the secondary linear classifier
    (1) X_train, X_val, X_test all contain vectors of consistent length where the predictions for an article are row vectors
    Position i of a row vector is therefore the prediction of the ith text segment of the article from the NB classifier
    To get vectors of consistent length they were post-padded with -1's (-1 does not correspond to any class IDs)

    (2) X_train_perc, X_val_perc, X_test_perc are matrices with predictions for an article corresponding to individual row vectors
    Position i of the row vector is the percentage of the article classified as belonging to class i by the NB classifier
    The length of the row vector is equal to the number of classes
    '''

    X_train = []
    y_train = []
    for t in train_id:
        tmp = np.where(text_id == t)[0]
        X_train.append(np.asarray(pred[tmp]))
        y_train.append(labels[tmp][0]) #labels on a per article basis

    X_val = []
    y_val = []
    for v in val_id:
        tmp = np.where(text_id == v)[0]
        X_val.append(pred[tmp])
        y_val.append(labels[tmp][0])

    X_test = []
    y_test = []
    for s in test_id:
        tmp = np.where(text_id == s)[0]
        X_test.append(pred[tmp])
        y_test.append(labels[tmp][0])

    # zero pad the X data so that everything has the same length
    len_x_train = max([len(i) for i in X_train])
    len_x_val = max([len(i) for i in X_val])
    len_x_test = max([len(i) for i in X_test])
    max_len = max(len_x_train, len_x_val, len_x_test)

    X_train = pad_sequences(X_train, maxlen=max_len, padding = 'post', value = -1)
    X_val = pad_sequences(X_val, maxlen = max_len, padding = 'post', value = -1)
    X_test = pad_sequences(X_test, maxlen = max_len, padding = 'post', value = -1)

    #create a second dataset which is the percentage of the article at each grade-level
    classes = np.unique(y_train)
    # convert the X sequences into percentages
    X_train_perc = np.zeros((len(y_train), len(classes)))
    num_samples = np.shape(X_train)[0]
    for i in range(num_samples):
        tmp = X_train[i, :]
        for n, item in enumerate(classes):
            ind = np.where(tmp == item)[0]
            frac = len(ind) / len(tmp)
            X_train_perc[i, n] = frac

    X_val_perc = np.zeros((len(y_val), len(classes)))
    num_samples = np.shape(X_val)[0]
    for i in range(num_samples):
        tmp = X_val[i, :]
        for n, item in enumerate(classes):
            ind = np.where(tmp == item)[0]
            frac = len(ind) / len(tmp)
            X_val_perc[i, n] = frac

    X_test_perc = np.zeros((len(y_test), len(classes)))
    num_samples = np.shape(X_test)[0]
    for i in range(num_samples):
        tmp = X_test[i, :]
        for n, item in enumerate(classes):
            ind = np.where(tmp == item)[0]
            frac = len(ind) / len(tmp)
            X_test_perc[i, n] = frac

    return X_train, X_val, X_test, X_train_perc, X_val_perc, X_test_perc, y_train, y_val, y_test

def calc_art_mean_median(X_train, y_train, X_val, y_val, num_classes):
    '''
    This function will calculate the mean article grade and median article grade based on the individual sentences.
    No training needed, only computed on the validation/test sets
    Ignore the zero padding
    :param X_train:
    :param X_val:
    :param X_test:
    :param y_train:
    :param y_val:
    :param y_test:
    :return:
    '''

    X_train = X_train.astype(float)
    X_train[X_train == -1] = np.nan
    pred_mean_tr = np.nanmean(X_train, axis = 1)
    pred_mean_tr = [int(round(i)) for i in pred_mean_tr]
    pred_median_tr = np.nanmedian(X_train, axis = 1)
    pred_median_tr = [int(round(i)) for i in pred_median_tr]

    X_val = X_val.astype(float)
    X_val[X_val == -1] = np.nan
    pred_mean_val = np.nanmean(X_val, axis=1)
    pred_mean_val = [int(round(i)) for i in pred_mean_val]
    pred_median_val = np.nanmedian(X_val, axis=1)
    pred_median_val = [int(round(i)) for i in pred_median_val]

    #calculate the accuracy for each
    acc_mean_tr = accuracy_score(y_train, pred_mean_tr)
    acc_median_tr = accuracy_score(y_train, pred_median_tr)
    acc_mean_val = accuracy_score(y_val, pred_mean_val)
    acc_median_val = accuracy_score(y_val, pred_median_val)

    #calculate the f1_score for each
    if num_classes == 2:
        f1_score_mean = f1_score(y_val, pred_mean_val)
        f1_score_median = f1_score(y_val, pred_median_val)
    else:
        f1_score_mean = f1_score(y_val, pred_mean_val, average = 'weighted')
        f1_score_median = f1_score(y_val, pred_mean_val, average = 'weighted')
    return pred_mean_val, pred_median_val, acc_mean_tr, acc_mean_val, f1_score_mean, acc_median_tr, acc_median_val, f1_score_median


def calc_linear_reg(X_train_perc, X_val_perc, y_train, y_val):
    '''
    Trains a linear regression classifier to make a prediction for an entire article based on the NB predictions for
    the article segments
    :param X_train_perc: training data with each sample being a vector of the percentage of an article predicted to
    be of a certain class according to the NB model
    :param X_val_perc: validation data
    :param y_train: labels for the training articles
    :param y_val: labels for the validation articles
    :return:
    '''

    classes = np.unique(y_train)

    reg = LinearRegression().fit(X_train_perc, y_train)
    y_pred = reg.predict(X_val_perc)
    y_pred = [int(round(i)) for i in y_pred]
    y_pred = np.asarray(y_pred)
    y_pred[y_pred > max(classes)] = max(classes)
    y_pred[y_pred < min(classes)] = min(classes)

    # calculate the training_error
    y_pred_tr = reg.predict(X_train_perc)
    y_pred_tr = [int(round(i)) for i in y_pred_tr]
    y_pred_tr = np.asarray(y_pred_tr)
    y_pred_tr[y_pred_tr > max(classes)] = max(classes)
    y_pred_tr[y_pred_tr < min(classes)] = min(classes)

    # calculate the accuracy for each
    acc_val = accuracy_score(y_val, y_pred)
    acc_train = accuracy_score(y_train, y_pred_tr)

    # calculate the f1_score for each
    if len(classes) == 2:
        f1_score_val = f1_score(y_val, y_pred)
    else:
        f1_score_val = f1_score(y_val, y_pred, average='weighted')
    return y_pred, acc_val, f1_score_val, acc_train

def calc_log_reg(X_train_perc, X_val_perc, y_train, y_val):
    '''
    Trains a logistic regression classifier to make a prediction for an entire article based on the NB predictions for
    the article segments
    :param X_train_perc: training data with each sample being a vector of the percentage of an article predicted to
    be of a certain class according to the NB model
    :param X_val_perc: validation data
    :param y_train: labels for the training articles
    :param y_val: labels for the validation articles
    :return:
    '''

    classes = np.unique(y_train)

    reg = LogisticRegression().fit(X_train_perc, y_train)
    y_pred = reg.predict(X_val_perc)
    y_pred = [int(round(i)) for i in y_pred]
    y_pred = np.asarray(y_pred)
    y_pred[y_pred > max(classes)] = max(classes)
    y_pred[y_pred < min(classes)] = min(classes)

    #calculate the training_error
    y_pred_tr = reg.predict(X_train_perc)
    y_pred_tr = [int(round(i)) for i in y_pred_tr]
    y_pred_tr = np.asarray(y_pred_tr)
    y_pred_tr[y_pred_tr > max(classes)] = max(classes)
    y_pred_tr[y_pred_tr < min(classes)] = min(classes)

    # calculate the accuracy for each
    acc_val = accuracy_score(y_val, y_pred)
    acc_train = accuracy_score(y_train, y_pred_tr)

    # calculate the f1_score for each
    if len(classes) == 2:
        f1_score_val = f1_score(y_val, y_pred)
    else:
        f1_score_val = f1_score(y_val, y_pred, average='weighted')
    return y_pred, acc_val, f1_score_val, acc_train


def top_5_words(model, words, n_grams, classes):
    '''
    Will return the top five predictands for every n-gram length used in the model
    For example, if the model has n-grams of length 1 and 2, then it will return the top five 1-word n-grams and top five
    2-word n-grams

    :param model: The NB model
    :param words: dictionary of all the words/n-grams used in the NB model
    :param n_grams: max length of n-grams used in the model
    :param classes: list of the classes
    :return:
    '''
    probs = model.feature_log_prob_

    n_classes, size_vocab = np.shape(probs)
    for n in range(n_classes):
        plt.plot(probs[n, :])
    plt.show()

    num_top = 5 #return the best 5 predictors
    top_words = []

    for i in range(n_classes):
        print('Grade {} has the following top predictands: '.format(classes[i]))
        top_ind = np.argsort(probs[i,:])
        top_ind = top_ind[::-1]
        best_words = words[top_ind]

        for j in range(len(n_grams[1])):
            print('N-gram length = {}:'.format(j))
            top_grams = [k for k in best_words if len(k.split(' ')) == j][:num_top]
            for t in top_grams:
                print(t)
        print('\n')
    return top_words

def run_NB_model(num_sentences, num_classes, max_n_gram, include_orig, texts, labels, text_id):
    '''
    Will train a NB model

    :param num_sentences: length of text segment to train on (in sentences), -1 implies the entire article
    :param num_classes: number of classes
    :param max_n_gram: maximum n-gram length to train on, will train with n-grams ranging from 1 to max_n_gram
    :param include_orig: flag on whether or not to include the original articles in the dataset (grade 12)
    :param texts: split texts based on num_sentences
    :param labels: labels corresponding to the split texts
    :param text_id: id that associates a text fragment to a particular article
    :return:
    '''

    n_grams = (1, max_n_gram)
    X_all, X_train, X_val, X_test, y_train, y_val, y_test, words, stop_words, classes, train_id, val_id, test_id = split_and_transform_data(texts, labels, text_id, n_grams)
    model = train_NB_model(X_train, y_train)
    pred_val = predict_from_NB_model(model, X_val)
    acc_val = accuracy_score(y_val, pred_val)
    pred_train = predict_from_NB_model(model, X_train)
    acc_train = accuracy_score(y_train, pred_train)
    if num_classes == 2:
        f1 = f1_score(y_val, pred_val)
    else:
        f1 = f1_score(y_val, pred_val, average = 'weighted')
    # return a dictionary summarizing the results of the model as well as the model parameters
    model_res = {}
    model_res['include_orig'] = include_orig
    model_res['num_sentences'] = num_sentences
    model_res['num_classes'] = num_classes
    model_res['max_n_gram'] = max_n_gram
    model_res['acc_NB_train'] = acc_train
    model_res['acc_NB_val'] = acc_val
    model_res['f1_NB'] = f1
    print(model_res)

    #compute the predictions for ALL of the X values (train, val, pred)
    all_pred = predict_from_NB_model(model, X_all)
    return model_res, pred_val, all_pred, train_id, val_id, test_id

def main():
    '''
    Cycles through varying combinations of sentences chunks, max n-gram lengths, binary and multi-class classification
    as well as including or not including the original articles

    Will train a NB classifier and associated linear classifier if trying on subsets of articles

    The linear classifiers include mean, median, linear regression and logistic regression

    Tracks all model runs to a dictionary which holds their training accuracy and test acc
    :return:
    '''

    num_sentences_range = [-1, 1, 2, 3]#[-1,1,2,3]
    num_classes_range = [2, 9]
    max_n_gram_range = [1, 2, 3]
    incl_ori_range = [True, False]

    model_runs = {}
    count = 0

    for o in incl_ori_range:
        for c in num_classes_range:
            for s in num_sentences_range:
                #create the dataset
                texts, labels, text_id = create_naive_bayes_datasets(s, c, o)
                for n in max_n_gram_range:
                    model_res, pred, all_pred, train_id, val_id, test_id = run_NB_model(s, c, n, o, texts, labels, text_id)
                    #create model stats
                    model_runs[count] = model_res
                    if s != -1:  #looking at a subset of an article - perform a second set of predictions
                        X_train, X_val, X_test, X_train_perc, X_val_perc, X_test_perc, y_train, y_val, y_test = create_article_dataset(all_pred, labels, text_id, train_id, val_id, test_id)
                        pred_mean_val, pred_median_val, acc_mean_tr, acc_mean_val, f1_score_mean, acc_median_tr, acc_median_val, f1_score_median = calc_art_mean_median(X_train, y_train, X_val, y_val, c)
                        # add to the dictionary for the model run
                        model_runs[count]['acc_mean_tr'] = acc_mean_tr
                        model_runs[count]['acc_mean_val'] = acc_mean_val
                        model_runs[count]['f1_mean'] = f1_score_mean
                        model_runs[count]['acc_median_tr'] = acc_median_tr
                        model_runs[count]['acc_median_val'] = acc_median_val
                        model_runs[count]['f1_median'] = f1_score_median

                        pred, acc_val, f1_score_val, acc_tr = calc_linear_reg(X_train_perc, X_val_perc, y_train, y_val)
                        model_runs[count]['acc_lin_tr'] = acc_tr
                        model_runs[count]['acc_lin_val'] = acc_val
                        model_runs[count]['f1_lin_val'] = f1_score_val

                        pred, acc_val, f1_score_val, acc_tr = calc_log_reg(X_train_perc, X_val_perc, y_train, y_val)
                        model_runs[count]['acc_log_tr'] = acc_tr
                        model_runs[count]['acc_log_val'] = acc_val
                        model_runs[count]['f1_log_val'] = f1_score_val
                    count += 1
    return model_runs

if __name__ == "__main__":
    model_runs = main()

    with open('model_runs.pickle', 'wb') as handle:
        pickle.dump(model_runs, handle, protocol=pickle.HIGHEST_PROTOCOL)
