'''
This script will be used for testing different combinations of regularizations as well as different
model configurations, utilizing glove word embeddings

Can be used for either binary or multi-class classification case

Will cycle through different combinations of L1 and L2 regularizations on the input weights, bias weights and
recurrent weights of the LSTM layer

Training history is saved to a pickle file

Must be run on GPU with Tensorflow 1.13

Author: Kiley Yeakel
'''

import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, CuDNNLSTM
import json
from tensorflow.keras.initializers import Constant
import pickle
from tensorflow.keras.regularizers import l1_l2


def load_word_embeddings(path):
    '''
    Loads in the word embedding file from the glove project
    :param path:
    :return: dictionary mapping word to associated vector
    '''
    embed = open(path)
    embed_lines = embed.read()
    #create the embedding matrix
    embed_lines = embed_lines.split('\n')
    embed_dict = {}
    for e in embed_lines:
        e = e.split(' ')
        e_word = e[0]
        e_vector = e[1:]
        embed_dict[e_word] = e_vector
    return embed_dict

def create_embed_matrix(word_index, embed_dict, embed_dim, vocab_size):
    '''
    creates an embedding matrix of dimension (num words in our dataset, dim embedding vector)
    :param word_index: index of words from our training examples
    :param embed_dict: pre-trained embedding from Glove project
    :return: all the embeddings for the words within our word_index
    '''

    sample_embedding = np.zeros((vocab_size, embed_dim))
    for i,word in enumerate(word_index):
        if word in embed_dict:
            sample_embedding[i, :] = embed_dict[word] #if we have an embedding for the word we put it in otherwise we have zeros
    return sample_embedding

def training_run(X_train, y_train, X_val, y_val, MAX_LEN, bias_reg, kernel_reg, recur_reg, word_index, class_assoc, X_test, y_test, num_epochs, embedding_layer):

    print('Training model.')

    # Train an LSTM - the model will only contain one layer
    # to add additional layers, simply add another CuDNNLSTM layer with option return_sequences = True
    # to use a different number of neurons change the first argument of the CuDNNLSTM layer
    model = Sequential()
    model.add(embedding_layer)
    model.add(CuDNNLSTM(100, bias_regularizer = bias_reg, kernel_regularizer = kernel_reg, recurrent_regularizer = recur_reg))
    if len(class_assoc) > 2: #multiclass classification
        model.add(Dense(len(class_assoc), activation='softmax'))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    else: #binary classification
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    hist = model.fit(X_train, y_train, epochs=num_epochs, batch_size=1000, validation_data=(X_val, y_val))
    hist = hist.history
    predictions = model.predict(X_test, batch_size=1000)
    return hist, predictions


def main():

    num_classes = 2
    #import the training data
    training_data = np.load('training_data_{}class.npz'.format(num_classes))
    X_train = training_data['x']
    y_train = training_data['y']
    MAX_LEN = np.shape(X_train)[1]
    print('Training data size = X: {}, y: {}'.format(np.shape(X_train), np.shape(y_train)))

    #import the validation data
    validation_data = np.load('validation_data_{}class.npz'.format(num_classes))
    X_val = validation_data['x']
    y_val = validation_data['y']

    #import the test data
    test_data = np.load('test_data_{}class.npz'.format(num_classes))
    X_test = test_data['x']
    y_test = test_data['y']

    #load in the word index
    with open('word_index_{}class.json'.format(num_classes), 'r') as fp:
        word_index = json.load(fp)

    #load in the class dictionary
    with open('class_assoc_{}class.json'.format(num_classes), 'r') as fp:
        class_assoc = json.load(fp)

    #create the embedding matrix
    embedding_dim = 50
    vocab_size = len(word_index) + 1
    pre_loaded_embed_path = os.path.join(os.getcwd(), 'Data/glove.6B/glove.6B.{}d.txt'.format(embedding_dim))
    embed_dict = load_word_embeddings(pre_loaded_embed_path)
    embed_mat = create_embed_matrix(word_index, embed_dict, embedding_dim, vocab_size)
    embedding_layer = Embedding(vocab_size, embedding_dim, embeddings_initializer=Constant(embed_mat),
                                input_length=MAX_LEN, trainable=True)

    Lreg_levels = [0, 0.01]
    num_epochs = 10
    reg_type = ['input','recur','bias']

    for r in reg_type:
        all_hist = []
        all_pred = []
        all_l1 = []
        all_l2 = []
        for l1 in Lreg_levels:
            for l2 in Lreg_levels:
                print('Training model with {} regularization of l1 = {} and l2 = {}'. format(r, l1, l2))
                reg = l1_l2(l1 = l1, l2 = l2)
                non_reg = l1_l2(l1 = 0, l2 = 0)

                if r =='bias':
                    hist, predictions = training_run(X_train, y_train, X_val, y_val, MAX_LEN, reg, non_reg, non_reg, word_index, class_assoc, X_test, y_test,num_epochs, embedding_layer)

                elif r == 'input':
                    hist, predictions = training_run(X_train, y_train, X_val, y_val, MAX_LEN, non_reg, reg, non_reg, word_index, class_assoc, X_test, y_test,num_epochs, embedding_layer)

                else:
                    hist, predictions = training_run(X_train, y_train, X_val, y_val, MAX_LEN, non_reg, non_reg, reg, word_index, class_assoc, X_test, y_test,num_epochs, embedding_layer)

                all_hist.append(hist)
                all_pred.append(predictions)
                all_l1.append(l1)
                all_l2.append(l2)

        np.savez('{}_reg_{}class_e{}.npz'.format(r,num_classes,embedding_dim), pred=all_pred)

        # create one dictionary that holds all of the training_data
        hist = {}
        acc = []
        val_acc = []
        loss = []
        val_loss = []
        for i in range(len(all_l1)):
            tmp = all_hist[i]
            acc.append(tmp['acc'])
            val_acc.append(tmp['val_acc'])
            loss.append(tmp['loss'])
            val_loss.append(tmp['val_loss'])
        hist['acc'] = acc
        hist['val_acc'] = val_acc
        hist['loss'] = loss
        hist['val_loss'] = val_loss

        with open('train_history_{}_reg_{}Class_e{}.pickle'.format(r, num_classes, embedding_dim), 'wb') as file:
            pickle.dump(hist, file)


if __name__ == '__main__':
    main()
