import math, random

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
        lambda1 = [0.16, 0.16, 0.17, 0.17, 0.17, 0.17]
        lambda2 = [0.27, 0.23, 0.2, 0.15, 0.1, 0.05]
        lambda3 = [0.05, 0.1, 0.15, 0.2, 0.23, 0.27]
        lambdas = self.set_lambdas(lambdas=lambda3)
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
        elif len(lambdas) != self.order + 1:
            return ValueError("Number of lambdas should be same as n + 1")
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
            models[code] = (create_ngram_model(NgramModel, "./train/" + code + ".txt"))
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

    print("Loading Data...")
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
    path = 'shakespeare_input.txt'
    model = create_ngram_model(NgramModelWithInterpolation, path, n=5, k=0.1)
    test_path = 'kanye_verses.txt'
    with open(test_path, encoding='utf-8', errors='ignore') as f:
        test_string = f.read()
    perp = model.perplexity(test_string)
    print("Perplexity: ", perp)
