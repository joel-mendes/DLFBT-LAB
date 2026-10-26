
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np
import time

def evaluate_model(model, X, y):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    scores = model.predict(X, verbose=0)
    predictions = np.argmax(scores, axis=1)
        
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_precision": float(
            precision_score(y, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y, predictions, average="weighted", zero_division=0)
        ),
        "predictions": predictions,
    }

    
def count_parameters(model):
    return model.count_params()


def measure_inference_time(model, X, y):
    model.predict(X[:1], verbose=0)

    start = time.time()
    model.predict(X, verbose=0)
    inference_time = time.time() - start

    return inference_time
