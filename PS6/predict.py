import torch
from PS6.main_classify import languages, predict
from PS6.models import CharRNNClassify

model = CharRNNClassify()
model.load_state_dict(torch.load('./model_classify.pth'))
model.eval()

test_dir = 'test.txt'
with open(test_dir, encoding='utf-8', errors='ignore') as file:
    test_words = file.read().strip().split('\n')

outputs = predict(model, test_words, None)
outputs = [x.item() for x in outputs]
print(outputs)

with open('labels.txt', 'w') as file:
    for pred in outputs:
        file.write(languages[pred] + '\n')
    file.close()
