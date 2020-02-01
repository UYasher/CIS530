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
        self.vocab = set()
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
                self.vocab.add(charac)

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        count_context = 0
        count_char = 0
        for ctx, charac in self.n_grams:
            if ctx == context:
                count_context += 1
                if charac == char:
                    count_char += 1
        if count_context == 0:
            return ArithmeticError("Division by zero")
        return count_char / count_context

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        vocab = self.get_vocab().sort()
        sum = 0
        for i in range(len(vocab)):
            sum += self.prob(context, vocab[i])
            if sum <= r:
                return vocab[i]

        return vocab[-1]


    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        generated_text = ""
        while length > 0:
            generated_text += self.random_char(generated_text)
            length -= 1

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        text = start_pad(self.n) + text

        sum_logs = 0
        for i in range(len(text)-self.n):
            sum_logs += math.log(self.prob(text[i:i+self.n], text[i+self.n]), 2)

        l = 1/len(text) * sum_logs

        return 2**-l


################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        pass

    def get_vocab(self):
        pass

    def update(self, text):
        pass

    def prob(self, context, char):
        pass

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

def load_dataset(folder):
    x = []
    y = []
    for code in COUNTRY_CODES:
        with open("./" + folder + "/" + code + ".txt", "r") as input_file:
            for city in input_file:
                x.append(input_file)
                y.append(code)
    return x, y


class AllCountriesModel():

    def __init__(self):
        models = {}
        for code in COUNTRY_CODES:
            models[code].append(create_ngram_model(NgramModel, "./train/" + code + ".txt"))
        self.models = models

    def predict_country(self, city):
        return max({ model.key:model.value(city) for model in self.models })

    def predict(self, cities):
        results = []
        for i in range(len(cities)):
            results.append(self.predict_country(cities[i]))

        return results

if __name__ == '__main__':
    x_train, y_train = load_dataset("train")
    x_dev, y_dev = load_dataset("val")

    model = AllCountriesModel()
    y_train_pred = model.predict(x_train)
    y_dev_pred = model.predict_country(x_dev)

    f1_train = f1_score(y_train, y_train_pred)
    confusion_train = confusion_matrix(y_train, y_train_pred)

    print("=====TRAINING=====")
    print("f1: " + str(f1_train))
    print(confusion_train)

    f1_test = f1_score(y_dev, y_dev_pred)
    confusion_test = confusion_matrix(y_dev, y_dev_pred)

    print("=====DEVELOPMENT=====")
    print("f1: " + str(f1_test))
    print(confusion_test)
