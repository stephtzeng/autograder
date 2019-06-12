# autograder
Automatic grade level complexity classifier

### Kneser-Ney Smoothing
To use `article_process.py`, you will need to install from: https://github.com/kpu/kenlm.

To instantiate the object, you will need the paths to:
- the folder with the text data
- the path to where you installed kenlm via git
- the subpath to the language models:
```python
path_to_data = 'path/to/data'
path_to_kenlm = 'path/to/github/kenlm'
path_to_arpa = path_to_kenlm + '/lm'
```

Usage:

```python
from article_lm import ArticleLM
articleLM = ArticleLM(path_to_data,
                      path_to_kenlm,
                      path_to_arpa, 5, 'grade_level', False)
```
where 5 denotes 5-gram, and you are creating grade level models, and original articles are not used.

## RNN
For trdaining the RNN:
--> run data_process.py
--> run train_model_gce_REG.py

Note that the RNN must be run on a GPU with Tensorflow 1.13 (other versions have different functions) and Keras.

The script will run through varying combinations of L1 and L2 regularizations on the input, bias and recurrent 
weights of the LSTM 1-layer model. Note that the script can be run on both binary and multiclass datasets. Also note that 
the model is fixed with only 1 LSTM layer with 100 neurons but this can easily be modified for more layers and neurons.
All run history is logged and saved.

## Naive Bayes
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
