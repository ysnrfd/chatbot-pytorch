import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random

# Step 1: Load and preprocess data
with open('intents.json', 'r') as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()

# Define stopwords
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        words = word_tokenize(pattern.lower())
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        all_words.extend(words)
        xy.append((words, intent['tag']))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# Lemmatize and remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Step 2: Create training data
X = []
y = []
for (pattern_words, tag) in xy:
    bag = [1 if w in pattern_words else 0 for w in all_words]
    X.append(bag)
    y.append(tags.index(tag))

X = np.array(X)
y = np.array(y)

# Step 3: Define PyTorch dataset and dataloader
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.X_data = torch.tensor(X, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset(X, y)
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

# Step 4: Define neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

# Step 5: Hyperparameters and training setup
input_size = len(X[0])
hidden_size = 32
output_size = len(tags)
learning_rate = 0.001
num_epochs = 2000

model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 6: Training the model
for epoch in range(num_epochs):
    for (inputs, targets) in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training complete.')

# Step 7: Save the model
torch.save(model.state_dict(), 'chatbot_model.pth')
print('Model saved.')

# Step 8: Function to predict intent from user input
def predict_intent(sentence, model, all_words, tags):
    model.eval()
    with torch.no_grad():
        words = word_tokenize(sentence.lower())
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        bag = [1 if w in words else 0 for w in all_words]
        input_data = torch.tensor(bag, dtype=torch.float32).unsqueeze(0)
        output = model(input_data)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        # Softmax to calculate probability
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        # Threshold to define confidence
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    return random.choice(intent['responses'])
        else:
            return "I'm sorry, I don't understand."

# Step 9: Interactive chat session
print("Start chatting with the chatbot (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    response = predict_intent(user_input, model, all_words, tags)
    print("Chatbot:", response)
