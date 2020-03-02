from PS6.load_names import all_categories, category_lines, n_letters, n_categories
from PS6.char_to_tensor import lineToTensor
from PS6.rnn import RNN
import math
import random
import time
import torch
import torch.nn as nn

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
criterion = nn.NLLLoss()
learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn

n_iters = 100000
print_every = 5000
plot_every = 1000

start = time.time()

# Keep track of losses for plotting
current_loss = 0
all_losses = []


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor):

    hidden = rnn.initHidden()
    rnn.zero_grad()

    outputs = None
    for i in range(line_tensor.size()[0]):
        outputs, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(outputs, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return outputs, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


for curr_iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if curr_iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (curr_iter, curr_iter / n_iters * 100, timeSince(start), loss,
                                                line, guess, correct))

    # Add current loss avg to list of losses
    if curr_iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
