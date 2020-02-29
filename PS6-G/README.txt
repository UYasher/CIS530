To generate text, do the following:

1) Load the model using:
model = RNN()
model.load_state_dict(torch.load(PATH))

2) Create a function to evaluate the model.
This can be done by importing evaluate from main_generate.py or writing similar code.

3) Assuming your evaluate function is of the same form as that in main_generate.py, run
evaluate('Th', 200, temperature=0.8)