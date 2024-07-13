import json
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Define your neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.l3(out)
        return out

# Load intents from JSON file
def load_intents(filename):
    with open(filename, 'r') as f:
        intents = json.load(f)
    return intents

# Tokenization and preprocessing of intents using basic bag-of-words
def preprocess_intents(intents, stemmer, ignore_words):
    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    # Stemming and preparing words
    all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = np.zeros(len(all_words), dtype=np.float32)
        pattern_words = [stemmer.stem(w.lower()) for w in pattern_sentence]
        for idx, w in enumerate(all_words):
            if w in pattern_words:
                bag[idx] = 1
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train, tags, all_words

# Dataset class for PyTorch DataLoader
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Main training function
def train_model(model, criterion, optimizer, train_loader, num_epochs=2000):
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words
            labels = labels.long()

            outputs = model(words)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Training complete.')

# Function to save trained model and associated data
def save_data(model, input_size, hidden_size, output_size, tags, all_words, filename):
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "tags": tags,
        "all_words": all_words
    }
    torch.save(data, filename)
    print(f'Training complete. Model and data saved to {filename}')

# Function to load trained model and associated data
def load_data(filename):
    data = torch.load(filename)
    return data["model_state"], data["input_size"], data["hidden_size"], data["output_size"], data["tags"], data["all_words"]

# Function for predicting class from user input using bag-of-words
def predict_class(model, sentence, tags, all_words, stemmer, device):
    model.eval()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1

    X = torch.from_numpy(bag).to(device)
    X = X.unsqueeze(0)  # Add batch dimension
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    return tag

# Function to get a random response for a given tag
def get_response(intents, tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that."

# Tokenization function
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Main execution
if __name__ == "__main__":
    # Define hyperparameters
    batch_size = 8
    input_size = 0  # To be determined by the number of unique words in all_words
    hidden_size = 8
    output_size = 0
    learning_rate = 0.001
    num_epochs = 1000
    filename = "intents.json"
    ignore_words = ['?', '!', '.', ',']

    # Initialize stemmer and load intents
    stemmer = PorterStemmer()
    intents = load_intents(filename)

    # Preprocess intents using basic bag-of-words
    X_train, y_train, tags, all_words = preprocess_intents(intents, stemmer, ignore_words)
    input_size = len(all_words)
    output_size = len(tags)

    # Create dataset and DataLoader
    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model, loss function, and optimizer
    model = NeuralNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, criterion, optimizer, train_loader, num_epochs)

    # Save trained model and associated data
    save_data(model, input_size, hidden_size, output_size, tags, all_words, "data.pth")

    # Load trained model and associated data
    model_state, input_size, hidden_size, output_size, tags, all_words = load_data("data.pth")

    # Initialize and load model for inference
    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Main interaction loop
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input('You: ')
        if sentence == "quit":
            break

        tag = predict_class(model, sentence, tags, all_words, stemmer, device)
        response = get_response(intents, tag)
        print(f"Bot: {response}")
