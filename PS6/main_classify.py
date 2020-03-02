import codecs
import math
import random
import string
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import torch.nn as nn
import matplotlib.pyplot as plt
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
    tensor = torch.zeros(len(line), 1, len(all_letters), dtype=torch.float)
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
    line, idx = random_choice(X)
    line_tensor = line_to_tensor(line)
    category = y[idx]
    one_hot = np.zeros(len(languages))
    one_hot[category] = 1
    category_tensor = torch.tensor(one_hot, dtype=torch.float)
    return category, line, category_tensor, line_tensor

'''
Input: trained model, a list of words, a list of class labels as integers
Output: a list of class labels as integers
'''
def predict(model, X, y):
    predictions = []
    hidden = model.init_hidden()
    for ii in range(len(X)):
        line_tensor = line_to_tensor(X[ii])
        line_tensor = line_tensor.permute(1, 0, 2)
        output, _ = model(line_tensor, hidden=hidden)
        output = torch.max(output, 1)[1]
        predictions.append(output[0])
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

    category, line, category_tensor, line_tensor = random_training_pair(X, y)

    hidden = model.init_hidden()

    optimizer.zero_grad()

    # for i in range(line_tensor.size()[0]):
    #     output, hidden = model(line_tensor[i], hidden)
    line_tensor = line_tensor.permute(1, 0, 2)
    output, _ = model(line_tensor, hidden=hidden)
    # print(output.size())
    category_tensor = category_tensor.view(1, len(languages))
    # loss = criterion(output, torch.max(category_tensor, 1)[1])
    loss = criterion(output, torch.max(category_tensor, 1)[1])
    loss.backward()

    # Should probably update with an optimizer instead of updating params manually

    # Add parameters' gradients to their values, multiplied by learning rate
    # for p in model.parameters():
    #     p.data.add_(-learning_rate, p.grad.data)
    optimizer.step()

    return output, loss.item(), category, line, model


'''
Use this to train and save your classification model. 
Save your model with the filename "model_classify"
'''
def run():
    # Init data
    X, y = readData("./")
    # print("X:")
    # print(X)
    # print()
    # print("y:")
    # print(y)

    # Init Network
    rnn = CharRNNClassify()

    # Init for training
    criterion = nn.NLLLoss()

    n_iters = 200000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    current_val_loss = 0
    all_losses = []
    all_val_losses = []

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    val_x, val_y = readData('./', train=False)

    for iter in range(1, n_iters + 1):

        output, loss, category, line, rnn = trainOneEpoch(rnn, criterion, optimizer, X, y)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Compute loss for random validation pair
        # _, _, val_category_tensor, val_line_tensor = random_training_pair(val_x, val_y)
        # val_line_tensor = val_line_tensor.permute(1, 0, 2)
        # val_output, _ = rnn(val_line_tensor, hidden=rnn.init_hidden())
        # val_category_tensor = val_category_tensor.view(1, len(languages))
        # val_loss = criterion(val_output, torch.max(val_category_tensor, 1)[1])
        # current_val_loss += val_loss

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            all_val_losses.append(current_val_loss / plot_every)
            current_loss = 0
            current_val_loss = 0

    acc = calculateAccuracy(rnn, val_x, val_y)
    print('Validation Accuracy: ', acc)

    torch.save(rnn.state_dict(), './model_classify.pth')
    # plt.figure()
    # plt.plot(all_losses, 'r', label='Train')
    # plt.plot(all_val_losses, 'b', label='Validate')
    # plt.title('Training/Validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend(loc='upper right')
    # num_points = int(n_iters / plot_every)
    # num_ticks = 5
    # spacing = int(num_points / num_ticks)
    # plt.xticks(np.arange(0, num_points + 1, spacing), [x * plot_every for x in range(0, num_points + 1, spacing)])
    # plt.show()


if __name__ == '__main__':
    run()
