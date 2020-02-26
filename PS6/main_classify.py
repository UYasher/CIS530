import codecs
import math
import random
import string
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from PS6.models import CharRNNClassify

'''
Don't change these constants for the classification task.
You may use different copies for the sentence generation model.
'''
languages = ["af", "cn", "de", "fi", "fr", "in", "ir", "pk", "za"]
all_letters = string.ascii_letters + " .,;'"

'''
Returns the words of the language specified by reading it from the data folder
Returns the validation data if train is false and the train data otherwise.
Return: A nx1 array containing the words of the specified language
'''
def getWords(baseDir, lang, train = True):
    folder = 'train/' if train else 'val/'
    with open(baseDir + folder + lang + '.txt', encoding='utf-8', errors='ignore') as file:
        line = file.read().strip().split('\n')
    return line

'''
Returns a label corresponding to the language
For example it returns an array of 0s for af
Return: A nx1 array as integers containing index of the specified language in the "languages" array
'''
def getLabels(lang, length):
    index = [i for i, country in enumerate(languages) if country == lang][0]
    labels = [index for _ in range(length)]
    return labels

'''
Returns all the languages and labels after reading it from the file
Returns the validation data if train is false and the train data otherwise.
You may assume that the files exist in baseDir and have the same names.
Return: X, y where X is nx1 and y is nx1
'''
def readData(baseDir, train=True):
    all_words = []
    all_labels = []
    for lang in languages:
        words = getWords(baseDir, lang, train=train)
        labels = getLabels(lang, len(words))
        all_words += words
        all_labels += labels
    return all_words, all_labels

'''
Convert a line/word to a pytorch tensor of numbers
Refer the tutorial in the spec
Return: A tensor corresponding to the given line
'''
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

'''
Returns the category/class of the output from the neural network
Input: Output of the neural networks (class probabilities)
Return: A tuple with (language, language_index)
        language: "af", "cn", etc.
        language_index: 0, 1, etc.
'''
def category_from_output(output):
    top_n, top_i = output.topk(1)
    language_index = top_i[0].item()
    language = languages[language_index]
    outputs = (language, language_index)
    return outputs

'''
Get a random input output pair to be used for training 
Refer the tutorial in the spec
'''
def random_training_pair(X, y):
    def random_choice(ls):
        idx = random.randint(0, len(ls)-1)
        return ls[idx], idx
    inputs, idx = random_choice(X)
    inputs = line_to_tensor(inputs)
    outputs = y[idx]
    return inputs, outputs

'''
Input: trained model, a list of words, a list of class labels as integers
Output: a list of class labels as integers
'''
def predict(model, X, y):
    predictions = []
    for i in range(len(X)):
        line_tensor = line_to_tensor(X[i])
        hidden = model.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)
        predictions.append(output)
    return predictions


'''
Input: trained model, a list of words, a list of class labels as integers
Output: The accuracy of the given model on the given input X and target y
'''
def calculateAccuracy(model, X, y):
    y_pred = predict(model, X, y)
    return accuracy_score(y, y_pred)

'''
Train the model for one epoch/one training word.
Ensure that it runs within 3 seconds.
Input: X and y are lists of words as strings and classes as integers respectively
Returns: You may return anything
'''
def trainOneEpoch(model, criterion, optimizer, X, y):
    learning_rate = 0.002

    for i in range(len(X)):

        category_tensor = y[i]
        line_tensor = line_to_tensor(X[i])

        hidden = model.initHidden()

        model.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in model.parameters():
            p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()


'''
Use this to train and save your classification model. 
Save your model with the filename "model_classify"
'''
def run():
    n_letters = len(all_letters)
    n_hidden = 128
    n_categories = len(languages)
    rnn = CharRNNClassify(n_letters)


# getWords('', 'af')
# x = getLabels('cn', 10)
# print(x)

