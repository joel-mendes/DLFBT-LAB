import torch
import torch.nn.functional as F
import torch.nn as nn

class DNNClassifier(nn.Module):
    def __init__(self, sequence_length, n_features, n_classes, hidden_size=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x):
        return self.net(x)

class RNNClassifier(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size=64, chunk_size=17, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.rnn = nn.RNN(chunk_size, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        if x.dim() == 3:            
            x = x.squeeze(-1)
        x = x.reshape(x.shape[0], -1, self.chunk_size) 
        _, h_n = self.rnn(x)
        return self.fc(self.bn(h_n[-1]))

class CNNClassifier(nn.Module):
    def __init__(self, input_size, n_classes, kernel_size=5, filters=32, **kwargs):
        super().__init__()
        padding = kernel_size // 2      # keeps the length unchanged for odd kernel sizes
        self.features = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(filters),
            nn.MaxPool1d(2),
            nn.Conv1d(filters, filters * 2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(filters * 2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(filters * 2, n_classes),
        )

    def forward(self, x):
        if x.dim() == 2:                
            x = x.unsqueeze(1)
        elif x.shape[-1] == 1:          
            x = x.transpose(1, 2)
        x = self.features(x).flatten(1)
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size=64, chunk_size=17, **kwargs):
        super().__init__()
        self.chunk_size = chunk_size
        self.lstm = nn.LSTM(chunk_size, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        x = x.reshape(x.shape[0], -1, self.chunk_size)
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.bn(h_n[-1]))