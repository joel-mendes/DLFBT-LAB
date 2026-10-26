import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, LSTM, GRU, SimpleRNN
from keras.layers import RepeatVector, Input


class DNNClassifier:
    def __init__(self, sequence_length, n_features, n_classes):
        self.sequence_length = sequence_length
        self.input_size = n_features
        self.num_classes = n_classes
        
        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_size,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            
            keras.layers.Dense(self.num_classes, activation="softmax"),
        ])
    
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model


class RNNClassifier:
    def __init__(self, input_size, n_classes):
        self.input_size = input_size
        self.num_classes = n_classes
        
        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_size, 1)),
            keras.layers.SimpleRNN(64, activation="tanh"),
            keras.layers.BatchNormalization(),

            #keras.layers.Dropout(0.3),
            #keras.layers.Dense(128, activation="relu"),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dropout(0.3),
            
            keras.layers.Dense(self.num_classes, activation="softmax"),
        ])

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            metrics=["accuracy"],
        )
        self.model = model
        
class CNNClassifier:
    def __init__(self, input_size, n_classes):
        self.input_size = input_size
        self.num_classes = n_classes

        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_size, 1)),  # (128, n_channels)
            keras.layers.Conv1D(64, kernel_size=5, padding="same",
                                activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dropout(0.3),

            #keras.layers.Conv1D(128, kernel_size=5, padding="same",
            #                    activation="relu"),
            #keras.layers.BatchNormalization(),
            #keras.layers.MaxPooling1D(pool_size=2),
            
            keras.layers.Dense(self.num_classes, activation="softmax"),
        ])
    
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model


class LSTMClassifier:
      def __init__(self, input_size, n_classes):
        self.input_size = input_size
        self.num_classes = n_classes
         
        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_size, 1)),  # (128, 9)
            keras.layers.LSTM(64),
            keras.layers.BatchNormalization(),
            #keras.layers.Dropout(0.3),
            
            keras.layers.Dense(self.num_classes, activation="softmax"),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model