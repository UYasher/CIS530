import nltk
from nltk.corpus import conll2002
from nltk import pos_tag
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support

# Assignment 7: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.
nltk.download('averaged_perceptron_tagger')


# Assigns features in tuples of 'feature name' and actual feature
def getfeats(word, o):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    o = str(o)
    features = [
        (o + 'word', word),
        # TODO: add more features here.
        # (o + 'pos', pos)
    ]
    return features


# Adds part-of-speech features
def getfeats_pos(pos, o):
    o = str(o)
    features = [
        (o + 'pos', pos)
    ]
    return features


# Retrieves features for the ith word (starting at 0) in the sentence 'sent'
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
            featlist = getfeats(word, o)
            featlist_pos = getfeats_pos(pos, o)
            features.extend(featlist)
            features.extend(featlist_pos)
    return dict(features)


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))
    
    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            feats = word2features(sent, i)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    print(len(train_feats))
    print(train_feats[:4])
    X_train = vectorizer.fit_transform(train_feats)
    print(X_train.shape)

    # TODO: play with other models
    model = Perceptron(verbose=1)
    model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # switch to test_sents for your final results
    for sent in dev_sents:
        for i in range(len(sent)):
            feats = word2features(sent,i)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results.txt")
    # format is: word gold pred
    with open("results.txt", "w") as out:
        for sent in dev_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python conlleval.py results.txt")






