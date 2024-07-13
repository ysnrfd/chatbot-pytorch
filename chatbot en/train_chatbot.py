import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize_words, bag_of_words, stem
from model import NeuralNet  # Assuming you have a model.py file with NeuralNet defined

# Load intents.json
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_patterns = []
tags = []
patterns_tags = []

# Extract patterns and tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        all_patterns.append(pattern)
        patterns_tags.append((pattern, intent['tag']))
    if intent['tag'] not in tags:
        tags.append(intent['tag'])

# Stem and tokenize words
ignore_words = ['?', '!', '.', ',']
all_words = [stem(word) for pattern in all_patterns for word in pattern.split() if word not in ignore_words]
all_words = sorted(set(all_words))  # Remove duplicates and sort

tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []
for (pattern, tag) in patterns_tags:
    bag = bag_of_words(pattern, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
batch_size = 8
learning_rate = 0.001
num_epochs = 1000

# Define dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'chatbot_model.pth')
print('Chatbot model trained and saved!')
