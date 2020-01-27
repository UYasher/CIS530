#############################################################
## ASSIGNMENT 2 CODE SKELETON
## RELEASED: 1/29/2019
## DUE: 2/5/2019
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
import numpy as np
from scipy.stats import zscore
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## A helper function for get_precision and get_recall which gives the values of the 2x2 confusion matrix
def get_2by2_confusion_matrix(y_pred, y_true):
    if not len(y_pred) == len(y_true):
        raise IndexError("y_pred and y_true are of different lengths")

    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(y_pred)):
        if y_true[i] == 1:
            if y_pred[i] == 1:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i] == 1:
                fp = 1
            else:
                tn += 1
    return tp, fn, fp, tn

## Calculates the precision of the predicted labels
def get_precision(y_pred, y_true):
    tp, fn, fp, tn = get_2by2_confusion_matrix(y_pred, y_true)
    precision = tp/(tp+fp)
    return precision
    
## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):
    tp, fn, fp, tn = get_2by2_confusion_matrix(y_pred, y_true)
    recall = tp/(tp+fn)
    return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = 2*precision*recall/(precision+recall)
    return fscore

## Prints out the precision, recall, and f-score
def test_predictions(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("fscore: " + str(fscore))
    return precision, recall, fscore

#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []   
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

### 2.1: A very simple baseline

########## Testing

## Makes feature matrix for all complex
def all_complex_feature(words):
    return [1 for _ in words]


## Labels every word complex
def all_complex(data_file):
    ## YOUR CODE HERE...
    words, labels = load_file(data_file)
    outputs = all_complex_feature(words)
    precision, recall, fscore = test_predictions(outputs, labels)
    performance = [precision, recall, fscore]
    return performance


### 2.2: Word length thresholding

## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):
    length_threshold_array = []
    for i in range(len(words)):
        if len(words[i]) >= threshold:
            length_threshold_array.append(1)
        else:
            length_threshold_array.append(0)

    return length_threshold_array

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file, threshold):  # Remove the threshold parameter after plotting
    twords, y_true_training = load_file(training_file)
    dwords, y_true_development = load_file(development_file)

    y_pred_training = length_threshold_feature(twords, threshold)
    y_pred_development = length_threshold_feature(dwords, threshold)

    tprecision = get_precision(y_pred_training, y_true_training)
    trecall = get_recall(y_pred_training, y_true_training)
    tfscore = get_fscore(y_pred_training, y_true_training)

    dprecision = get_precision(y_pred_development, y_true_development)
    drecall = get_recall(y_pred_development, y_true_development)
    dfscore = get_fscore(y_pred_development, y_true_development)


    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


## Make feature matrix consisting of word lengths
def length_feature(words):
    lengths = []
    for i in range(len(words)):
        lengths.append(len(words[i]))
    return lengths

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file): 
   counts = defaultdict(int) 
   with gzip.open(ngram_counts_file, 'rt', encoding='UTF-8') as f:
       for line in f:
           token, count = line.strip().split('\t') 
           if token[0].islower(): 
               counts[token] = int(count) 
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set


# Computes raw frequencies
def frequency_features(words, counts):
    frequencies = []
    for word in words:
        frequencies.append(counts[word])
    return frequencies

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    output = []
    for word in words:
        output.append(1) if counts[word] > threshold else output.append(0)
    return output

def word_frequency_threshold(training_file, development_file, counts, threshold):  # Remove the threshold parameter after plotti
    ## YOUR CODE HERE
    twords, tlabels = load_file(training_file)
    dwords, dlabels = load_file(development_file)
    toutputs = frequency_threshold_feature(twords, threshold, counts)
    doutputs = frequency_threshold_feature(dwords, threshold, counts)
    print("=====Training=====")
    tprecision, trecall, tfscore = test_predictions(toutputs, tlabels)
    print("=====Development=====")
    dprecision, drecall, dfscore = test_predictions(doutputs, dlabels)
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


# Plots Precision-Recall Curve over different thresholds for either word length or frequency baselines
def plot_curve_baseline(training_file, development_file, counts, thresholds, use_length):
    train_recall = []
    train_prec = []
    dev_recall = []
    dev_prec = []
    for th in thresholds:
        if use_length:
            train, dev = word_length_threshold(training_file, development_file, th)
        else:
            train, dev = word_frequency_threshold(training_file, development_file, counts, th)
        train_recall.append(train[1])
        train_prec.append(train[0])
        dev_recall.append(dev[1])
        dev_prec.append(dev[0])
    fig, axes = plt.subplots(1, 2)
    feat = 'length' if use_length else 'frequency'
    fig.suptitle('Precision-Recall Curve for word %s from thresholds %d to %d' % (feat, thresholds[0], thresholds[-1]))
    axes[0].plot(train_recall, train_prec, 'r')
    axes[0].title.set_text('Training data')
    axes[1].plot(dev_recall, dev_prec, 'b')
    axes[1].title.set_text('Development data')
    for ii in range(2):
        axes[ii].set_xlabel('Recall')
        axes[ii].set_ylabel('Precision')
    plt.show()

### 2.4: Naive Bayes

def get_standard_features(words, frequency_threshold, counts):
    length_array = length_feature(words)

    scaler = StandardScaler()
    scaler.fit(scaler)
    scaled_length_array = scaler.transform(length_array)

    features = np.array([
        words,
        scaled_length_array,
        frequency_threshold_feature(words, frequency_threshold, counts)
    ])

    return features


## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    twords, y_true_training = load_file(training_file)
    dwords, y_true_development = load_file(development_file)

    # X_train = get_standard_features(twords, 9, counts)
    X_train, train_stats = features_train(twords, counts)
    clf = GaussianNB()
    clf.fit(X_train, y_true_training)

    X_dev = features_test(dwords, counts, train_stats)
    y_pred_training = clf.predict(X_train)
    y_pred_development = clf.predict(X_dev)

    print("=====Training=====")
    tprecision, trecall, tfscore = test_predictions(y_pred_training, y_true_training)
    print("=====Development=====")
    dprecision, drecall, dfscore = test_predictions(y_pred_development, y_true_development)

    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance


### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):
    twords, y_true_training = load_file(training_file)
    dwords, y_true_development = load_file(development_file)

    # X_train = get_standard_features(twords, 9, counts)
    X_train, train_stats = features_train(twords, counts)
    clf = LogisticRegression()
    clf.fit(X_train, y_true_training)

    X_dev = features_test(dwords, counts, train_stats)
    y_pred_training = clf.predict(X_train)
    y_pred_development = clf.predict(X_dev)

    print("=====Training=====")
    tprecision, trecall, tfscore = test_predictions(y_pred_training, y_true_training)
    print("=====Development=====")
    dprecision, drecall, dfscore = test_predictions(y_pred_development, y_true_development)

    training_performance = (tprecision, trecall, tfscore)
    development_performance = (dprecision, drecall, dfscore)
    return development_performance

### 2.7: Build your own classifier


# Feature extraction method for training data
def features_train(words, counts):
    length_feats = np.array(length_feature(words))
    length_mean = np.mean(length_feats, axis=0)
    length_std = np.std(length_feats, axis=0)
    freq_feats = np.array(frequency_features(words, counts))
    freq_mean = np.mean(freq_feats, axis=0)
    freq_std = np.std(freq_feats, axis=0)
    # Normalize features
    length_feats = np.array(zscore(length_feats))
    freq_feats = np.array(zscore(freq_feats))
    return np.c_[length_feats, freq_feats], [length_mean, length_std, freq_mean, freq_std]


# Feature extraction method for testing (development) data
def features_test(words, counts, train_stats):
    length_feats = np.array(length_feature(words))
    length_feats = (length_feats - train_stats[0]) / train_stats[1]
    freq_feats = np.array(frequency_features(words, counts))
    freq_feats = (freq_feats - train_stats[2]) / train_stats[3]
    return np.c_[length_feats, freq_feats]


## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"

    train_data = load_file(training_file)
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    plot_curve_baseline(training_file, development_file, counts, thresholds=np.arange(5, 15), use_length=True)

    # naive_bayes_results = naive_bayes(training_file, development_file, counts)
    # logistic_regression_results = logistic_regression(training_file, development_file, counts)
    # print(naive_bayes_results)
    # print(logistic_regression_results)
