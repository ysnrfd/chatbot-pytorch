# chat.py

import random
import json
import torch
from nltk_utils import tokenize_words, stem
from model import NeuralNet

# Load intents from intents.json
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Load all_words from all_words.json
with open('all_words.json', 'r') as file:
    all_words = json.load(file)

# Define input_size, hidden_size, and output_size
input_size = len(all_words)
hidden_size1 = 16  # Adjust based on your model
hidden_size2 = 8   # Adjust based on your model
output_size = len(intents['intents'])

# Load the trained model with strict=False
model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size)
model.load_state_dict(torch.load('chatbot_model.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

# Define functions for inference
def predict_class(sentence, model):
    tokenized_words = tokenize_words(sentence)
    stemmed_words = [stem(word) for word in tokenized_words]
    bag_of_words = [0]*len(all_words)
    for idx, w in enumerate(all_words):
        if w in stemmed_words:
            bag_of_words[idx] = 1

    input_data = torch.tensor(bag_of_words, dtype=torch.float32).unsqueeze(0)
    output = model(input_data)
    _, predicted = torch.max(output, dim=1)
    tag = intents['intents'][predicted.item()]['tag']

    return tag

def get_response(intents_list, intents_json):
    tag = predict_class(sentence, model)
    if tag not in intents_list:
        return "Sorry, I didn't get that. Can you please repeat?"

    responses = intents_json['intents'][intents_list.index(tag)]['responses']
    return random.choice(responses)

# Chatbot interface
print("Chat with me (type 'quit' to exit):")
while True:
    sentence = input("You: ")
    if sentence.lower() == 'quit':
        break

    response = get_response([intent['tag'] for intent in intents['intents']], intents)
    print("Bot:", response)
