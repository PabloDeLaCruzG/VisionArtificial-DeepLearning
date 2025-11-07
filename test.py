## quitar warnings
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

## TAREA 1 creo pero tambien sirve para la 2 y 3 creo (REVISAR BIEN) es el codigo de https://keras.io/examples/vision/mnist_convnet/

import keras
from keras import layers
import numpy as np

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = (
    keras.datasets.mnist.load_data()
)  # HABRIA QUE CAMBIAR mnist por Cifar10

# Model / data parameters
num_classes = 10
input_shape = x_train[0].shape

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Flatten())
model.add(layers.Dense(48, activation="sigmoid"))  # capa oculta
model.add(layers.Dense(num_classes, activation="softmax"))

model.summary()

batch_size = 32
epochs = 10

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Contar tiempo de inicio
import time

start_time = time.time()

# Entrenar el modelo
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Contar tiempo de fin
end_time = time.time()
# Calcular el tiempo de ejecución
execution_time = end_time - start_time

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Tiempo de ejecución:", execution_time, "segundos")
