
'''
def fit(*args, **kwargs):
    raise NotImplementedError("fit() has not been implemented yet.")
'''
#Avoid writing three completely independent copies of the same training procedure.
import time # once completed put the import time in the main and delete this one
# so its not loading everytime 

def fit(model, X_train, y_train, X_val, y_val, **kwargs):
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        **kwargs
    )
    training_time = time.time() - start
    return {
        "model": model,
        "history": {
            "train_loss": history.history.get("loss", []),
            "val_loss": history.history.get("val_loss", []),
            "train_accuracy": history.history.get("accuracy", []),
            "val_accuracy": history.history.get("val_accuracy", []),
        },
        "training_time": training_time,
        "epochs_completed": len(history.history.get("loss", [])),
    }