import torch
from PS6.main_classify import languages, predict, random_training_pair, readData
from PS6.models import CharRNNClassify
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

model = CharRNNClassify()
model.load_state_dict(torch.load('./model_classify.pth'))
model.eval()

n_categories = 9
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

val_x, val_y = readData('./', train=False)
for i in range(n_confusion):
    category_i, _, val_tensor, val_line_tensor = random_training_pair(val_x, val_y)
    val_line_tensor = val_line_tensor.permute(1, 0, 2)
    val_output, _ = model(val_line_tensor, hidden=model.init_hidden())
    val_category_tensor = val_tensor.view(1, len(languages))
    guess_i = torch.max(val_output, 1)[1][0]
    confusion[category_i][guess_i] += 1

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + languages, rotation=90)
ax.set_yticklabels([''] + languages)

# Force label at every tick
ax.set_xlabel('Predicted labels')
ax.xaxis.set_label_position('top')
ax.set_ylabel('True labels')
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()
