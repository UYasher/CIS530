import math, random
from sklearn.metrics import f1_score, confusion_matrix

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
        self.n_grams = ngrams(n, '')
        self.vocab = list()
        pass

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        new_grams = ngrams(self.order, text)
        for context, charac in new_grams:
            self.n_grams.append((context, charac))
            if charac not in self.vocab:
                self.vocab.append(charac)

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        count_context = 0
        count_char = 0
        input(len(self.n_grams))
        for ctx, charac in self.n_grams:
            if ctx == context:
                count_context += 1
                if charac == char:
                    count_char += 1
        return (count_char + 1) / (count_context + len(self.vocab))

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        vocab = self.get_vocab()
        vocab.sort()
        sum = 0
        for i in range(len(vocab)):
            # print("r: " + str(r))
            # input("sum: " + str(sum))
            sum += self.prob(context, vocab[i])
            sum += 0.1
            if sum >= r:
                print(vocab[i])
                return vocab[i]
        return vocab[-1]

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        generated_text = ""
        while length > 0:
            generated_text += self.random_char(generated_text[-self.order:])
            # print(generated_text)
            length -= 1
        return generated_text

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

    def __init__(self):
        models = {}
        for code in COUNTRY_CODES:
            models[code] = (create_ngram_model(NgramModel, "./train/" + code + ".txt", k=1))
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

    def predict(self, cities):
        results = []
        for i in range(len(cities)):
            results.append(self.predict_country(cities[i]))

        return results

if __name__ == '__main__':

    # print("Loading Data...")
    # x_train, y_train = load_dataset("train")
    # x_dev, y_dev = load_dataset("val")
    #
    # print("Training Model...")
    # model = AllCountriesModel()
    #
    # print("Making Predictions...")
    # y_train_pred = model.predict(x_train)
    # y_dev_pred = model.predict_country(x_dev)
    #
    # print("Tabulating Results...")
    # f1_train = f1_score(y_train, y_train_pred)
    # confusion_train = confusion_matrix(y_train, y_train_pred)
    #
    # print("=====TRAINING=====")
    # print("f1: " + str(f1_train))
    # print(confusion_train)
    #
    # f1_test = f1_score(y_dev, y_dev_pred)
    # confusion_test = confusion_matrix(y_dev, y_dev_pred)
    #
    # print("=====DEVELOPMENT=====")
    # print("f1: " + str(f1_test))
    # print(confusion_test)

    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', n=2)
    print(m.random_text(250))
