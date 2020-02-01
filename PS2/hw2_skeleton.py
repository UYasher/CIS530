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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
import matplotlib.pyplot as plt
from itertools import combinations


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
                fp += 1
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


def load_test_file(data_file):
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
            i += 1
    return words

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

## Make feature matrix for word_frequency_threshold
def frequency_threshold_feature(words, threshold, counts):
    output = []
    for word in words:
        output.append(0) if counts[word] > threshold else output.append(1)
    return output

def word_frequency_threshold(training_file, development_file, counts, threshold):  # Remove the threshold parameter after plotti
    ## YOUR CODE HERE
    twords, tlabels = load_file(training_file)
    dwords, dlabels = load_file(development_file)
    toutputs = frequency_threshold_feature(twords, threshold, counts)
    doutputs = frequency_threshold_feature(dwords, threshold, counts)
    # print("=====Training=====")
    tprecision, trecall, tfscore = test_predictions(toutputs, tlabels)
    # print("=====Development=====")
    dprecision, drecall, dfscore = test_predictions(doutputs, dlabels)
    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance


# Plots Precision-Recall Curve over different thresholds for either word length or frequency baselines
def plot_curve_baseline(training_file, development_file, counts, thresholds, use_length):
    train_recall = []
    train_prec = []
    train_f1 = []
    dev_recall = []
    dev_prec = []
    dev_f1 = []
    for th in thresholds:
        if use_length:
            train, dev = word_length_threshold(training_file, development_file, th)
        else:
            train, dev = word_frequency_threshold(training_file, development_file, counts, th)
        train_recall.append(train[1])
        train_prec.append(train[0])
        train_f1.append(train[2])
        dev_recall.append(dev[1])
        dev_prec.append(dev[0])
        dev_f1.append(dev[2])
    train_idx = train_f1.index(max(train_f1))
    dev_idx = dev_f1.index(max(dev_f1))
    # print('----------------')
    # print('Threshold: ', thresholds[train_idx])
    # print(train_prec[train_idx], train_recall[train_idx], train_f1[train_idx])
    # print(dev_prec[train_idx], dev_recall[train_idx], dev_f1[train_idx])
    # print(dev_prec[dev_idx], dev_recall[dev_idx], dev_f1[dev_idx])
    fig, axes = plt.subplots(1, 2)
    feat = 'length' if use_length else 'frequency'
    axes[0].plot(train_recall, train_prec, 'r')
    axes[0].title.set_text('Training data')
    axes[1].plot(dev_recall, dev_prec, 'b')
    axes[1].title.set_text('Development data')
    fig.suptitle('Precision-Recall Curve for word %s' % feat, fontweight='bold', y=1.0)
    for ii in range(2):
        axes[ii].set_xlabel('Recall')
        axes[ii].set_ylabel('Precision')
    plt.show()


# Baseline feature extraction method for training data
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


### 2.4: Naive Bayes

## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):
    twords, y_true_training = load_file(training_file)
    dwords, y_true_development = load_file(development_file)

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

# Normalization methods for classifiers

def features_train_custom(words, counts):
    length_feats = np.array(length_feature(words))
    length_mean = np.mean(length_feats, axis=0)
    length_std = np.std(length_feats, axis=0)
    freq_feats = np.array(frequency_features(words, counts))
    freq_mean = np.mean(freq_feats, axis=0)
    freq_std = np.std(freq_feats, axis=0)
    vowel_feats = np.array(vowel_features(words))
    vowel_mean = np.mean(vowel_feats, axis=0)
    vowel_std = np.std(vowel_feats, axis=0)
    sy_mean, sy_max = syllable_length(words)
    sy_mean, sy_max = np.array(sy_mean), np.array(sy_max)
    sy_mean_mean = np.mean(sy_mean, axis=0)
    sy_mean_std = np.std(sy_mean, axis=0)
    sy_max_mean = np.mean(sy_max, axis=0)
    sy_max_std = np.std(sy_max, axis=0)
    rare_feats = np.array(rare_members(words))
    rare_mean = np.mean(rare_feats, axis=0)
    rare_std = np.std(rare_feats, axis=0)
    mode_feats = np.array(mode_count(words))
    mode_mean = np.mean(mode_feats, axis=0)
    mode_std = np.std(mode_feats, axis=0)
    # Normalize features
    length_feats = np.array(zscore(length_feats))
    freq_feats = np.array(zscore(freq_feats))
    vowel_feats = np.array(zscore(vowel_feats))
    sy_mean = np.array(zscore(sy_mean))
    sy_max = np.array(zscore(sy_max))
    rare_feats = np.array(zscore(rare_feats))
    mode_feats = np.array(zscore(mode_feats))
    return np.c_[length_feats, freq_feats, vowel_feats, sy_mean, sy_max, rare_feats, mode_feats], \
           [length_mean, length_std, freq_mean, freq_std, vowel_mean, vowel_std, sy_mean_mean, sy_mean_std,
            sy_max_mean, sy_max_std, rare_mean, rare_std, mode_mean, mode_std]


# Feature extraction method for testing (development) data
def features_test_custom(words, counts, train_stats):
    length_feats = np.array(length_feature(words))
    length_feats = (length_feats - train_stats[0]) / train_stats[1]
    freq_feats = np.array(frequency_features(words, counts))
    freq_feats = (freq_feats - train_stats[2]) / train_stats[3]
    vowel_feats = np.array(vowel_features(words))
    vowel_feats = (vowel_feats - train_stats[4]) / train_stats[5]
    sy_mean = np.array(syllable_length(words)[0])
    sy_mean = (sy_mean - train_stats[6]) / train_stats[7]
    sy_max = np.array(syllable_length(words)[1])
    sy_max = (sy_max - train_stats[8]) / train_stats[9]
    rare_feats = np.array(rare_members(words))
    rare_feats = (rare_feats - train_stats[10]) / train_stats[11]
    mode_feats = np.array(mode_count(words))
    mode_feats = (mode_feats - train_stats[12]) / train_stats[13]
    return np.c_[length_feats, freq_feats, vowel_feats, sy_mean, sy_max, rare_feats, mode_feats]

# Features

# Make feature matrix consisting of word lengths
def length_feature(words):
    lengths = []
    for i in range(len(words)):
        lengths.append(len(words[i]))
    return lengths


# Computes raw frequencies
def frequency_features(words, counts):
    frequencies = []
    for word in words:
        frequencies.append(counts[word])
    return frequencies


# Computes ratio of vowels within each word
def vowel_features(words):
    vowels = []
    vowel_list = ['a', 'e', 'i', 'o', 'u']
    for word in words:
        char_list = [1 for char in word if char in vowel_list]
        vowels.append(len(char_list) / len(word))
    return vowels


# Computes average length between syllables in each word
def syllable_length(words):
    sy_lengths_mean = []
    sy_lengths_max = []
    vowel_list = ['a', 'e', 'i', 'o', 'u']
    for word in words:
        word_sy = []
        cnt = 0
        for idx, char in enumerate(word):
            if char in vowel_list:
                word_sy.append(cnt)
                cnt = 0
            elif idx == len(word) - 1:
                cnt += 1
                word_sy.append(cnt)
            else:
                cnt += 1
        sy_lengths_mean.append(np.mean(np.array(word_sy), axis=0))
        sy_lengths_max.append(max(word_sy))
    return sy_lengths_mean, sy_lengths_max


# Computes occurrences of rare alphabets within each word
def rare_members(words):
    rare_outputs = [0 for _ in words]
    rare_chars = ['j', 'q', 'x', 'z']
    for idx, word in enumerate(words):
        for char in word:
            if char in rare_chars:
                rare_outputs[idx] += 1
    return rare_outputs


# Computes how many times the most frequent letter appears within each word
def mode_count(words):
    mode_outputs = [0 for _ in words]
    for idx, word in enumerate(words):
        maxchar = max(set(word), key=word.count)
        mode_outputs[idx] += word.count(maxchar) - 1
    return mode_outputs


# Prints examples of TP/TN/FP/FN words in the training data
def error_analysis(words, y_pred, y_true):
    cnt_tp, cnt_tn, cnt_fp, cnt_fn = 0, 0, 0, 0
    words_tp, words_tn, words_fp, words_fn = [], [], [], []
    for idx, word in enumerate(words):
        if y_true[idx] == 1:
            if y_pred[idx] == 1:
                if cnt_tp < 10:
                    words_tp.append(word)
                cnt_tp += 1
            else:
                if cnt_fn < 10:
                    words_fn.append(word)
                cnt_fn += 1
        else:
            if y_pred[idx] == 1:
                if cnt_fp < 10:
                    words_fp.append(word)
                cnt_fp += 1
            else:
                if cnt_tn < 10:
                    words_tn.append(word)
                cnt_tn += 1
    print("True Positive: ", words_tp)
    print("True Negative: ", words_tn)
    print("False Positive: ", words_fp)
    print("False Negative: ", words_fn)


## Trains a classifier of your choosing, predicts labels for the test dataset
## and writes the predicted labels to the text file 'test_labels.txt',
## with ONE LABEL PER LINE

def test_all_classifiers(training_file, development_file, counts, test=False):
    twords, y_true_training = load_file(training_file)
    if test:
        dwords = load_test_file(development_file)
    else:
        dwords, y_true_development = load_file(development_file)

    X_train, train_stats = features_train_custom(twords, counts)
    X_dev = features_test_custom(dwords, counts, train_stats)

    X_train = X_train[:,
              [j for j in range(np.size(X_train, 1)) if "11110111"[j] == "1"]
              ]
    X_dev = X_dev[:,
            [j for j in range(np.size(X_dev, 1)) if "11110111"[j] == "1"]
            ]

    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto', probability=True)))
    models.append(('RF', RandomForestClassifier()))
    models.append(('ADA', AdaBoostClassifier()))
    models.append(('GB', GradientBoostingClassifier()))
    models.append(('XGB', XGBClassifier()))
    models.append(('XGBRF', XGBRFClassifier()))
    models.append(('MLP', MLPClassifier()))

    for idx, model in enumerate(models):
        model[1].fit(X_train, y_true_training)
        y_pred_training = model[1].predict(X_train)
        y_pred_development = model[1].predict(X_dev)
        print(model[0])
        print("=====Training=====")
        tprecision, trecall, tfscore = test_predictions(y_pred_training, y_true_training)
        if not test:
            print("=====Development=====")
            dprecision, drecall, dfscore = test_predictions(y_pred_development, y_true_development)

        if model[0] == 'MLP':
            #models.append(('Voting', VotingClassifier(list(models), voting="soft")))
            models.append(('Voting2', VotingClassifier(list(models[7:12]), voting="soft")))

        if model[0] == "XGB":
           return y_pred_development


def test_all_classifiers_all_feature_sets(training_file, development_file, counts, test=False):
    twords, y_true_training = load_file(training_file)
    if test:
        dwords = load_test_file(development_file)
    else:
        dwords, y_true_development = load_file(development_file)

    X_train_full, train_stats = features_train_custom(twords, counts)
    X_dev_full = features_test_custom(dwords, counts, train_stats)

    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto', probability=True)))
    models.append(('RF', RandomForestClassifier()))
    models.append(('ADA', AdaBoostClassifier()))
    models.append(('GB', GradientBoostingClassifier()))
    models.append(('XGB', XGBClassifier()))
    models.append(('XGBRF', XGBRFClassifier()))
    models.append(('MLP', MLPClassifier()))

    results = []

    #input(np.size(X_train_full, 1))

    for i in range(2**np.size(X_train_full, 1) - 1):
            feature_set = str(bin(i+1))[2:]
            if 6 >  feature_set.count("1") >= 5:
                while len(feature_set) < np.size(X_train_full, 1) - 1:
                    feature_set = "0" + feature_set
                feature_set = "1" + feature_set
                print(feature_set)
                print([j for j in range(np.size(X_train_full, 1)) if feature_set[j] == "1"])
                X_train = X_train_full[:,
                    [j for j in range(np.size(X_train_full, 1)) if feature_set[j] == "1"]
                ]
                X_dev = X_dev_full[:,
                    [j for j in range(np.size(X_dev_full, 1)) if feature_set[j] == "1"]
                ]
                print(X_train.shape)

                for model in models:
                    model[1].fit(X_train, y_true_training)
                    y_pred_training = model[1].predict(X_train)
                    y_pred_development = model[1].predict(X_dev)
                    print(model[0])
                    print("=====Training=====")
                    tprecision, trecall, tfscore = test_predictions(y_pred_training, y_true_training)
                    if not test:
                        print("=====Development=====")
                        dprecision, drecall, dfscore = test_predictions(y_pred_development, y_true_development)

                    if model[0] == 'MLP':
                        #models.append(('Voting', VotingClassifier(list(models), voting="soft")))
                        models.append(('Voting2', VotingClassifier(list(models[7:12]), voting="soft")))

                    results.append([feature_set, model[0], tfscore, dfscore])

    return results


def train_MLP(training_file, development_file, counts, test=False):
    twords, y_true_training = load_file(training_file)
    if test:
        dwords = load_test_file(development_file)
    else:
        dwords, y_true_development = load_file(development_file)

    X_train, train_stats = features_train_custom(twords, counts)
    X_dev = features_test_custom(dwords, counts, train_stats)

    models = []
    mlp = MLPClassifier()
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    models.append(('MLP', GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5, refit='F1')))

    for idx, model in enumerate(models):
        model[1].fit(X_train, y_true_training)
        y_pred_training = model[1].predict(X_train)
        y_pred_development = model[1].predict(X_dev)
        print(model[0])
        print("=====Training=====")
        tprecision, trecall, tfscore = test_predictions(y_pred_training, y_true_training)
        if not test:
            print("=====Development=====")
            dprecision, drecall, dfscore = test_predictions(y_pred_development, y_true_development)

        # Best parameter set
        print('Best parameters found:\n', models[0][1].best_params_)

        # All results
        means = models[0][1].cv_results_['mean_test_score']
        stds = models[0][1].cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, models[0][1].cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        return y_pred_development


if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"
    
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    # t, d = word_frequency_threshold(training_file, development_file, counts, 1e8)
    # plot_curve_baseline(training_file, development_file, counts, thresholds=np.arange(1, 13), use_length=True)
    # plot_curve_baseline(training_file, development_file, counts, thresholds=np.arange(1e6, 1e8 + 1, 1e5), use_length=False)

    # naive_bayes_results = naive_bayes(training_file, development_file, counts)
    # logistic_regression_results = logistic_regression(training_file, development_file, counts)
    # print(naive_bayes_results)
    # print(logistic_regression_results)

    predictions = test_all_classifiers(training_file, development_file, counts, test=False)
    #predictions = test_all_classifiers_all_feature_sets(training_file, development_file, counts, test=False)
    #predictions = test_all_classifiers(training_file, test_file, counts, test=True)
    #predictions = train_MLP(training_file, development_file, counts, test=True)
    with open('test_labels.txt', 'a') as file:
        for pred in predictions:
            file.write(str(pred) + '\n')
