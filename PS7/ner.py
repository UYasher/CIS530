import nltk
import pickle
import re
from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import TruncatedSVD
import sklearn_crfsuite
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


# Assigns features in tuples of 'feature name' and actual feature
def getfeats(word, pos, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    features = [
        (o + 'word', word),
        # TODO: add more features here.
        (o + 'pos', pos),
        (o + 'lower', word.lower()),
        (o + 'upper', word.isupper()),
        # (o + 'name', spanish_names(word)),
        # (o + 'place', spanish_loc(word)),
        (o + 'digit', contains_digits(word)),
        (o + '2prefix', word[:2]),
        (o + '3prefix', word[:3]),
        (o + '4prefix', word[:4]),
        (o + '2suffix', word[-2:]),
        (o + '3suffix', word[-3:]),
        (o + '4suffix', word[-4:]),
        # (o + 'word_shape', word_shape(word)),
        # (o + 'hyphen', contains_hyphen(word)),
        # ((o + 'chunk'), syntactic_chunk(word)),
    ]
    return features


# Determines whether the given word is a Spanish name
def spanish_names(word):
    return word in names


# Determines whether the given word is a Spanish location
def spanish_loc(word):
    return word in places


# Retrieves features for the ith word (starting at 0) in the sentence 'sent'
# TODO: Might also pass in sent to getfeats so we can chunk
def word2features(sent, i):
    """ The function generates all features
    for the word at position i in the
    sentence."""
    features = []
    # the window around the token
    window_size = 3
    for o in range(-1 * window_size, window_size + 1):
        if 0 <= i + o < len(sent):
            word = sent[i+o][0]
            pos = sent[i+o][1]
            featlist = getfeats(word, pos, o)
            features.extend(featlist)
    return dict(features)


def contains_digits(word):
    return any(char.isdigit for char in word)


def contains_hyphen(word):
    return '-' in word


def word_shape(word):
    word = re.sub('[A-Z]', 'X', word)
    word = re.sub('[a-z]', 'x', word)
    word = re.sub('[0-9]', 'd', word)
    return word


def syntactic_chunk(sent):
    return nltk.ne_chunk(sent)

if __name__ == "__main__":
    # Load a list of Spanish names
    # name_file = 'esp_names.txt'
    # names = list()
    # with open(name_file, 'r') as file:
    #     for line in file:
    #         names.extend(line.split())

    # Load a list of Spanish places
    # loc_file = 'esp_places.txt'
    # places = list()
    # with open(loc_file, 'r') as file:
    #     for line in file:
    #         places.extend(line.split())

    # Load the training data
    print("Loading training data...")
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    print("Processing training data...")
    train_feats = []
    train_labels = []

    crf_train_feats = []
    crf_train_labels = []
    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent, i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])
        crf_train_feats.append(train_feats)
        crf_train_labels.append(train_labels)

    print("Vectorizing data...")
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # print("Scaling data...")
    # max_abs_scaler = preprocessing.MaxAbsScaler()
    # X_train = max_abs_scaler.fit_transform(X_train)
    # print(X_train)

    # num_comp = 150

    print("Training model...")
    # TODO: play with other models
    # model = Perceptron(verbose=1)
    # model = LogisticRegression(verbose=1)
    # model = SVC(verbose=2)
    # model = GradientBoostingClassifier(verbose=1, max_depth=1)
    model = AdaBoostClassifier()
    # model = sklearn_crfsuite.CRF(algorithm='lbfgs', c1=0.1, c2=0.1, all_possible_transitions=True, verbose=1)
    # X_train = TruncatedSVD(n_components=num_comp).fit_transform(X_train)
    # model = MLPClassifier(hidden_layer_sizes=50, verbose=1, max_iter=150)
    model.fit(X_train, train_labels)
    pickle.dump(model, open('model', 'wb'))
    # model.fit(crf_train_feats, crf_train_labels)

    test_feats = []
    test_labels = []

    crf_test_feats = []
    crf_test_labels = []
    # switch to test_sents for your final results
    # for sent in test_sents:
    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent, i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])
        crf_test_feats.append(test_feats)
        crf_test_labels.append(test_labels)

    X_test = vectorizer.transform(test_feats)
    # X_test = TruncatedSVD(n_components=num_comp).fit_transform(X_test)
    y_pred = model.predict(X_test)
    # y_pred = model.predict(crf_test_feats)

    j = 0
    print("Writing results...")
    # format is: word gold pred
    with open("ada_train_results.txt", "w") as out:
        for sent in train_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")
