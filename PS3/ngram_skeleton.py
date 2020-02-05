import math, random
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix
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

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
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
        return self.vocab

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
        return (count_char + self.smoothing) / (count_context + self.smoothing * len(self.vocab))

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        vocab = self.get_vocab()
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
            sum_logs += math.log(self.prob(text[i:i+self.order], text[i+self.order]), 2)

        out = 1/len(text) * sum_logs

        return 2**-out


################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        super(NgramModelWithInterpolation, self).__init__(n, k)
        self.n_grams_all = []
        for i in range(1, n + 1):
            self.n_grams_all.append(NgramModel(i, k))

    def get_vocab(self):
        return self.n_grams_all[0].get_vocab()

    def update(self, text):
        for i in range(0, self.order):
            self.n_grams_all[i].update(text)

    def prob(self, context, char):
        lambdas = self.set_lambdas()
        output_prob = 0
        for ii in range(len(lambdas)):
            output_prob += lambdas(ii) * self.n_grams_all[ii].prob(context, char)
        return output_prob

    # Helper function for setting lambda values to be used in the interpolation
    def set_lambdas(self, lambdas=None):
        if lambdas is None:
            lambdas = []
            if self.order == 0:
                return [1]
            for ii in range(self.order):
                lambdas.append(1 / self.order)
        elif len(lambdas) != self.order:
            return ValueError("Number of lambdas should be same as n")
        elif sum(lambdas) != 1:
            return ValueError("Sum of lambdas should equal to 1")
        return lambdas


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


class AllCountriesModel():

    def __init__(self, n, k):
        models = {}
        for code in COUNTRY_CODES:
            # models[code] = (create_ngram_model(NgramModel, "./train/" + code + ".txt", n=n, k=k))
            models[code] = (create_ngram_model(NgramModelWithInterpolation, "./train/" + code + ".txt", n=n, k=k))
        self.models = models

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

    def set_lambdas(self, lambdas):
        for code in self.models:
            print(self.models[code].set_lambdas(lambdas)

    def fit(self, cities):
        results = []
        for i in range(len(cities)):
            results.append(self.predict_country(cities[i]))

        return results




if __name__ == '__main__':

    # fscores = np.array(
    #     [[0, 0.4933333333333334, 0.4933333333333334, 0.4922222222222222, 0.4922222222222222, 0.4911111111111111, 0.49,
    #       0.49, 0.4911111111111111, 0.4888888888888889],
    #      [0, 0.6333333333333333, 0.6366666666666667, 0.6377777777777778, 0.6388888888888888, 0.64, 0.6422222222222222,
    #       0.6455555555555555, 0.6488888888888888, 0.6466666666666666],
    #      [0, 0.6311111111111111, 0.6344444444444445, 0.6311111111111111, 0.6355555555555555, 0.63, 0.6255555555555555,
    #       0.62, 0.6088888888888889, 0.6022222222222222],
    #      [0, 0.5844444444444444, 0.5566666666666666, 0.5188888888888888, 0.4955555555555556, 0.4677777777777778,
    #       0.4444444444444444, 0.4211111111111111, 0.40444444444444444, 0.39111111111111113],
    #      [0, 0.4177777777777778, 0.3522222222222222, 0.3188888888888889, 0.29333333333333333, 0.27555555555555555, 0.26,
    #       0.2411111111111111, 0.21666666666666667, 0.20555555555555555],
    #      [0, 0.27555555555555555, 0.21666666666666667, 0.19, 0.17888888888888888, 0.16444444444444445,
    #       0.15555555555555556, 0.14777777777777779, 0.14333333333333334, 0.14],
    #      [0, 0.18333333333333332, 0.15888888888888889, 0.1411111111111111, 0.13555555555555557, 0.13111111111111112,
    #       0.12666666666666668, 0.12444444444444444, 0.12333333333333335, 0.12]]
    # )
    #
    # plt.imshow(fscores, cmap='YlGnBu', interpolation='nearest', vmin=0, vmax=1, )
    # plt.colorbar()
    # plt.ylabel("n")
    # plt.xlabel("k")
    # plt.title("F-Scores")
    # plt.show()

    print("Loading Data...")
    x_train, y_train = load_dataset("train")
    x_dev, y_dev = load_dataset("val")


    print("Training Model...")
    n = 1
    k = 8
    lambdas_list = [
        [1, 0],
        [0, 1]
    ]

    model = AllCountriesModel(n=n, k=k)

    for lambdas in lambdas_list:
        print(lambdas)

        model.set_lambdas(lambdas)

        # print("Making Predictions...")
        y_train_pred = model.fit(x_train)
        y_dev_pred = model.fit(x_dev)

        # print("Tabulating Results...")
        f1_train = f1_score(y_train, y_train_pred, average="micro")
        confusion_train = confusion_matrix(y_train, y_train_pred)

        print("=====TRAINING=====")
        print("f1: " + str(f1_train))
        print(confusion_train)

        f1_dev = f1_score(y_dev, y_dev_pred, average="micro")
        confusion_dev = confusion_matrix(y_dev, y_dev_pred)

        print("=====DEVELOPMENT=====")
        print("f1: " + str(f1_dev))
        print(confusion_dev)

    # n_k_arr[n].append(f1_dev)

    # m = create_ngram_model(NgramModel, 'shakespeare_input.txt', n=2)
    # print(m.random_text(250))
    # print(n_k_arr)
    # print(max(map(max, n_k_arr)))

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