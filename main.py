import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten for MLP models
x_train_flatten = x_train.reshape(len(x_train), 28*28)
x_test_flatten = x_test.reshape(len(x_test), 28*28)

kaggle_test = pd.read_csv('test.csv')

# Reshape & Normalize Kaggle test data
kaggle_test = kaggle_test.values.reshape(-1, 28, 28) / 255.0

# Function to save predictions in Kaggle format
def save_predictions(predictions, filename):
    submission = pd.DataFrame({"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions})
    submission.to_csv(filename, index=False)
    print(f"Saved {filename}")

# (i)
slpmodel = keras.Sequential([keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')])
slpmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
slpmodel.fit(x_train_flatten, y_train, epochs=5)
slp_predictions = np.argmax(slpmodel.predict(kaggle_test.reshape(len(kaggle_test), 784)), axis=1)
save_predictions(slp_predictions, "slp_submission.csv")

# (ii)
mlpmodel = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,), activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
mlpmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlpmodel.fit(x_train_flatten, y_train, epochs=5)
mlp_predictions = np.argmax(mlpmodel.predict(kaggle_test.reshape(len(kaggle_test), 784)), axis=1)
save_predictions(mlp_predictions, "mlp_submission.csv")

# (iii)
cnn_model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)
cnn_predictions = np.argmax(cnn_model.predict(kaggle_test.reshape(-1, 28, 28, 1)), axis=1)
save_predictions(cnn_predictions, "cnn_submission.csv")

print("All submissions saved!")
