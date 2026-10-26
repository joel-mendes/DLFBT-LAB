import tensorflow as tf

def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
