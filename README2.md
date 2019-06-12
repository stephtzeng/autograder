# autograder
Automatic grade level complexity classifier

For training the RNN:
--> run data_process.py
--> run train_model_gce_REG.py

Note that the RNN must be run on a GPU with Tensorflow 1.13 (other versions have different functions) and Keras.

The script will run through varying combinations of L1 and L2 regularizations on the input, bias and recurrent 
weights of the LSTM 1-layer model. Note that the script can be run on both binary and multiclass datasets. Also note that 
the model is fixed with only 1 LSTM layer with 100 neurons but this can easily be modified for more layers and neurons.
All run history is logged and saved.

For training the Naive Bayes (NB) classifier plus added linear classifier:
--> NB_classifier_v5.py

This script will run through several different combinations of settings including:
- numbers of sentences in each text segment chunk
- maxmimum number of n-grams
- inclusion (or not) of the original articles
- binary and multi-class (grade-level) classification

For each combination, if the text segments are smaller than an entire article, the script will train a secondary linear classifier.
The linear classifiers take the NB predictions for the individual segments and aggregate them into a prediction for the 
entire articles. The linear classifiers explored include:
- mean
- median
- logistic regression
- linear regression
For each combination, the train accuracy, test accuracy, and test f1 score are reported. All results are saved to file.
