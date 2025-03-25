import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() 

# Normalizing the dataset 
x_train = x_train/255
x_test = x_test/255
  
# Flatting the dataset in order 
# to compute for model building 
x_train_flatten = x_train.reshape(len(x_train), 28*28) 
x_test_flatten = x_test.reshape(len(x_test), 28*28) 

#i)
slpmodel = keras.Sequential([keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')]) 
slpmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
slpmodel.fit(x_train_flatten, y_train, epochs=5) 
print("Evaluating Single Layer Perceptron: ")
slpmodel.evaluate(x_test_flatten, y_test) 

#ii)
mlpmodel = keras.Sequential([keras.layers.Dense(128, input_shape=(784,), activation='relu'),  # Hidden layer 1
    keras.layers.Dense(64, activation='relu'),                     # Hidden layer 2
    keras.layers.Dense(10, activation='softmax')                   # Output layer
])

mlpmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlpmodel.fit(x_train_flatten, y_train, epochs=5)
print("Evaluating Multi Layer Perceptron: ")
mlpmodel.evaluate(x_test_flatten, y_test)

#iii)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"Test accuracy: {test_acc:.4f}")

