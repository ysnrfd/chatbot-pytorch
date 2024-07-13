import json
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import random
# Uncomment the line below if nltk data is not already downloaded
# nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

# Load intents from JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Preprocess data
all_words = []
tags = []
xy = []

# Process each intent
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # Tokenize each pattern
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Define set of ignore words
ignore_words = ['?', '!', '.', ',']
# Stem and lower each word, excluding ignore words
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
# Sort words and remove duplicates
all_words = sorted(set(all_words))

# Sort tags and remove duplicates
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # Create bag of words
    bag = np.zeros(len(all_words), dtype=np.float32)
    pattern_words = [stemmer.stem(w.lower()) for w in pattern_sentence]
    for idx, w in enumerate(all_words):
        if w in pattern_words:
            bag[idx] = 1
    X_train.append(bag)
    # Encode tags
    label = tags.index(tag)
    y_train.append(label)

# Convert lists to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Define dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.tensor(X_train, dtype=torch.float32)
        self.y_data = torch.tensor(y_train, dtype=torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# Create dataset and dataloader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Define neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Initialize model, loss function, and optimizer
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(words)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training complete.')

# Save trained model and related data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'Model saved to {FILE}')

# Inference
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stemmer.stem(w.lower()) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag

def predict_class(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    X = X.to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    return tag

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']
            return random.choice(responses)

print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    tag = predict_class(sentence)
    response = get_response(tag)
    print(f"Bot: {response}")
