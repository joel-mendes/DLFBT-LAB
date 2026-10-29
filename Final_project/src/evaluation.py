import time

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .training import get_device


def predict(model, X, batch_size=256, device=None):
    """Return the predicted class index for every row of X (numpy array)."""
    device = device or get_device()
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(device)
            predictions.append(model(batch).argmax(dim=1).cpu())
    return torch.cat(predictions).numpy()


def evaluate_model(model, X, y, batch_size=256, device=None):
    predictions = predict(model, X, batch_size, device)

    return {
        "accuracy": accuracy_score(y, predictions),
        "macro_precision": precision_score(y, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(y, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(y, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, predictions, average="weighted", zero_division=0),
        "predictions": predictions,
    }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, X, batch_size=256, device=None, repeats=5):
    """Average seconds needed to predict all of X."""
    predict(model, X, batch_size, device)  # warm-up run, not timed

    times = []
    for _ in range(repeats):
        start = time.time()
        predict(model, X, batch_size, device)
        times.append(time.time() - start)
    return sum(times) / len(times)