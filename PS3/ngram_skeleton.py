import math, random
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np


################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    text = start_pad(n) + text
    ngram_list = []
    for i in range(len(text)-n):
        ngram_list.append((text[i:i+n], text[i+n]))
    return ngram_list

def create_ngram_model(model_class, path, n=2, k=0.0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0.0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.order = n
        self.smoothing = k
        self.n_grams = dict()
        self.vocab = list()
        pass

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return set(self.vocab)

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        new_grams = ngrams(self.order, text)
        for context, charac in new_grams:
            if context in self.n_grams:
                if charac in self.n_grams[context]:
                    self.n_grams[context][charac] += 1
                else:
                    self.n_grams[context][charac] = 1
                self.n_grams[context]['sum'] += 1
            else:
                self.n_grams[context] = {charac: 1, 'sum': 1}
            if charac not in self.vocab:
                self.vocab.append(charac)

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        count_context = 0
        count_char = 0
        if context in self.n_grams:
            count_context = self.n_grams[context]['sum']
            if char in self.n_grams[context]:
                count_char = self.n_grams[context][char]
        if count_context == 0:
            return 1 / len(self.vocab)
        return (count_char + self.smoothing) / (count_context + self.smoothing * len(self.vocab))

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        vocab = self.vocab
        vocab.sort()
        sum = 0
        for i in range(len(vocab)):
            sum += self.prob(context, vocab[i])
            if sum > r:
                return vocab[i]
        return vocab[-1]

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        generated_text = start_pad(self.order)
        while length > 0:
            generated_text += self.random_char(generated_text[-self.order:])
            length -= 1
        return generated_text[self.order::]

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        text = start_pad(self.order) + text

        sum_logs = 0
        for i in range(len(text)-self.order):
            curr_prob = self.prob(text[i:i+self.order], text[i+self.order])
            if curr_prob == 0:
                return float('inf')
            sum_logs += math.log(curr_prob, 2)

        out = 1/(len(text) - self.order) * sum_logs

        return 2**-out


################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        super(NgramModelWithInterpolation, self).__init__(n, k)
        self.n_grams_all = []
        for i in range(0, n + 1):
            self.n_grams_all.append(NgramModel(i, k))

    def get_vocab(self):
        return set(self.n_grams_all[-1].get_vocab())

    def update(self, text):
        for i in range(0, self.order + 1):
            self.n_grams_all[i].update(text)

    def prob(self, context, char):
        lambdas = self.set_lambdas()
        lambdas = self.lambdas
        output_prob = 0
        for ii in range(len(lambdas)):
            level_context = context[-ii:] if ii > 0 else ''
            output_prob += lambdas[ii] * self.n_grams_all[ii].prob(level_context, char)
        return output_prob

    # Helper function for setting lambda values to be used in the interpolation
    def set_lambdas(self, lambdas=None):
        if lambdas is None:
            lambdas = []
            if self.order == 0:
                return [1]
            for ii in range(self.order + 1):
                lambdas.append(1 / (self.order + 1))
        elif len(lambdas) != self.order:
            return ValueError("Number of lambdas should be same as n")
        elif sum(lambdas) != 1:
            return ValueError("Sum of lambdas should equal to 1")
        return lambdas

    def update_lambdas(self, lambdas):
        self.lambdas = lambdas

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

def load_dataset(folder):
    x = []
    y = []
    for code in COUNTRY_CODES:
        with open("./" + folder + "/" + code + ".txt", encoding='utf-8', errors='ignore') as input_file:
            for city in input_file:
                x.append(city.strip())
                y.append(code)
    return x, y


class AllCountriesModel:

    def __init__(self, n, k, interpolation=False):
        models = {}
        for code in COUNTRY_CODES:
            if interpolation:
                models[code] = (create_ngram_model(NgramModelWithInterpolation, "./train/" + code + ".txt", n=n, k=k))
            else:
                models[code] = (create_ngram_model(NgramModel, "./train/" + code + ".txt", n=n, k=k))
        self.models = models
        self.n = n

    def predict_country(self, city):
        max_prob = 0
        arg_max = ""
        for code in COUNTRY_CODES:
            order = self.models[code].order
            padded_city = start_pad(order) + str(city)
            probability = 1
            for i in range(len(city)):
                probability *= self.models[code].prob(padded_city[i:i+order], padded_city[i+order])
            if probability > max_prob:
                max_prob = probability
                arg_max = code
        return arg_max

    def update_lambdas(self, lambdas):
        for code in self.models:
            self.models[code].update_lambdas(lambdas)

    def fit(self, cities):
        results = []
        for i in range(len(cities)):
            results.append(self.predict_country(cities[i]))

        return results


def n_k_gridsearch(x_train, y_train, x_dev, y_dev, min_n=0, min_k=1, max_n=6, max_k=10):

    n_k_arr_train = []
    n_k_arr_dev = []

    for n in range(0, max_n):
        print("n = " + str(n))
        n_k_arr_train.append([0])
        n_k_arr_dev.append([0])
        for k in range(1, max_k):
            print("k = " + str(k))
            model = AllCountriesModel(n=n, k=k)

            y_train_pred = model.fit(x_train)
            y_dev_pred = model.fit(x_dev)

            f1_train = f1_score(y_train, y_train_pred, average="micro")
            f1_dev = f1_score(y_dev, y_dev_pred, average="micro")

            n_k_arr_train[n].append(f1_train)
            n_k_arr_dev[n].append(f1_dev)

    return n_k_arr_train, n_k_arr_dev


def lambdas_gridsearch(x_train, y_train, x_dev, y_dev, model, steps=11):

    train_performance_dict = {}
    dev_performance_dict = {}

    lambdas_list = [lambdas for lambdas in
                     filter(
                         lambda ntuple: sum(ntuple) == 1,
                        [ntuple for ntuple in itertools.product( np.linspace(0, 1, steps), repeat=model.n)]
                     )
                    ]

    for lambdas in lambdas_list:
        print(lambdas)
        model.update_lambdas(lambdas)

        y_train_pred = model.fit(x_train)
        y_dev_pred = model.fit(x_dev)

        f1_train = f1_score(y_train, y_train_pred, average="micro")
        f1_dev = f1_score(y_dev, y_dev_pred, average="micro")
        print(f1_dev)

        train_performance_dict[lambdas] = f1_train
        dev_performance_dict[lambdas] = f1_dev

    return train_performance_dict, dev_performance_dict


if __name__ == '__main__':

    print("Loading Data...")
    x_train, y_train = load_dataset("train")
    x_dev, y_dev = load_dataset("val")


    # print("Finding Optimal n and k...")
    # fscores_train, fscores_dev = n_k_gridsearch(x_train, y_train, x_dev, y_dev)
    # fscores_train = np.array(fscores_train)
    # fscores_dev = np.array(fscores_dev)
    # print("Development F Scores for various n,k -- ")
    # print(fscores_dev)
    # print("Max of")
    # print(max(map(max, fscores_dev)))
    # print("@")
    # print(np.argmax(fscores_dev))
    # print("Additionally, here are the training F Scores --")
    # print(fscores_train)

    # print("Plotting n,k heatmap for dev set...")
    # plt.imshow(fscores_dev, cmap='YlGnBu', interpolation='nearest', vmin=0, vmax=1, )
    # plt.colorbar()
    # plt.ylabel("n")
    # plt.xlabel("k")
    # plt.title("Development Set F-Scores")
    # plt.show()

    # print("Plotting n,k heatmap for test set...")
    # plt.imshow(fscores_train, cmap='YlGnBu', interpolation='nearest', vmin=0, vmax=1, )
    # plt.colorbar()
    # plt.ylabel("n")
    # plt.xlabel("k")
    # plt.title("Training Set F-Scores")
    # plt.show()

    n = 3
    k = 3
    print("n=" + str(n) + ", k=" + str(k))
    model = AllCountriesModel(n=n, k=k, interpolation=True)

    # print("Calculating Optimal Lambdas...")
    # fscores_train_dict, fscores_dev_dict = lambdas_gridsearch(x_train, y_train, x_dev, y_dev, model, steps=5)
    # print(fscores_dev_dict)

    lambdas = (0.1, 0.5, 0.4)
    model.update_lambdas(lambdas)

    y_train_pred = model.fit(x_train)
    y_dev_pred = model.fit(x_dev)

    print("=====TRAINING=====")
    f1_train = f1_score(y_train, y_train_pred, average="micro")
    confusion_train = confusion_matrix(y_train, y_train_pred)
    print("f1: " + str(f1_train))
    print(confusion_train)

    print("=====DEVELOPMENT=====")
    f1_dev = f1_score(y_dev, y_dev_pred, average="micro")
    confusion_dev = confusion_matrix(y_dev, y_dev_pred)
    print("f1: " + str(f1_dev))
    print(confusion_dev)

    print("=====COMBINING FOR TEST SET=====")
    x_combined = x_train + x_dev
    y_combined = y_train + y_dev
    model = AllCountriesModel(n=n, k=k, interpolation=True)
    model.update_lambdas(lambdas)

    x_test = []
    with open("cities_test.txt", encoding='utf-8', errors='ignore') as input_file:
        for line in input_file:
            x_test.append(line.strip())

    y_test_pred = model.fit(x_test)

    with open("test_labels.txt", "w") as output_file:
        for line in y_test_pred:
            output_file.write(line+"\n")

    print()
    print("Test set predictions can be found in test_labels.txt")

    for i in range(len(x_dev)):
        print(x_dev[i])
        print(model.fit([x_dev[i]]))
        print(y_dev[i])

    # path = 'shakespeare_input.txt'
    # model = create_ngram_model(NgramModel, path, n=2, k=0.1)
    # test_path = 'shakespeare_lines_processed.txt'
    # with open(test_path, encoding='utf-8', errors='ignore') as f:
    #     test_string = f.read()
    # perp = model.perplexity(test_string)
    # print("Perplexity: ", perp)