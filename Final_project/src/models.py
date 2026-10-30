import torch
import torch.nn.functional as F
import torch.nn as nn

class DNNClassifier(nn.Module):
    def __init__(self, sequence_length, n_features, n_classes, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, X):
        X = torch.relu(self.fc1(X.flatten(1)))
        return self.fc2(X)


class RNNClassifier(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, X):
        X = X.reshape(-1, self.input_size, 1)
        _, h_n = self.rnn(X)
        return self.fc(h_n[-1])


class CNNClassifier(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size=32, kernel_size=3):
        super().__init__()
        self.input_size = input_size
        self.conv = nn.Conv1d(1, hidden_size, kernel_size, padding=kernel_size // 2)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, X):
        X = X.reshape(-1, 1, self.input_size)
        X = torch.relu(self.conv(X))
        return self.fc(X.mean(-1))


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size=64):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, X):
        X = X.reshape(-1, self.input_size, 1) 
        _, (h_n, _) = self.lstm(X)
        return self.fc(h_n[-1])


