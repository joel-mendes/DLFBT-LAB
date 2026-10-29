import torch
import torch.nn.functional as F
import torch.nn as nn

class DNNClassifier(nn.Module):
    def __init__(self, sequence_length, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_classes),   # raw logits, no softmax
        )

    def forward(self, x):
        return self.net(x)

class RNNClassifier(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size=64, **kwargs):
        super().__init__()
        # input_size is the sequence length; each time step has 1 feature
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        if x.dim() == 2:                # (batch, seq_len) -> (batch, seq_len, 1)
            x = x.unsqueeze(-1)
        _, h_n = self.rnn(x)            # h_n: (1, batch, hidden)
        return self.fc(self.bn(h_n[-1]))  # raw logits


class CNNClassifier(nn.Module):
    def __init__(self, input_size, n_classes, **kwargs):
        super().__init__()
        # input_size is the sequence length; the signal has 1 channel
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),    # (batch, 64, 1), independent of length
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),   # raw logits, no softmax
        )

    def forward(self, x):
        if x.dim() == 2:                # (batch, seq_len) -> (batch, 1, seq_len)
            x = x.unsqueeze(1)
        elif x.shape[-1] == 1:          # (batch, seq_len, 1) -> (batch, 1, seq_len)
            x = x.transpose(1, 2)
        x = self.features(x).flatten(1)
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, n_classes, hidden_size=64, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        if x.dim() == 2:                # (batch, seq_len) -> (batch, seq_len, 1)
            x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)      # h_n: (1, batch, hidden)
        return self.fc(self.bn(h_n[-1]))  # raw logits