# model.py

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_p=0.2):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(dropout_p)
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(dropout_p)
        self.l3 = nn.Linear(hidden_size2, 30)  # Adjust to match the saved model
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.dropout1(out)
        out = self.relu(self.l2(out))
        out = self.dropout2(out)
        out = self.l3(out)
        return out
