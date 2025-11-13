from random import sample
import tensorflow as ts
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

import keras
from keras import layers
from keras import ops

# Funciones auxiliares para visualizar los resultados de los modelos entrenados
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time  # Para medir el tiempo de entrenamiento

# =============================================================================
# 1. CONFIGURACIÓN GLOBAL
# =============================================================================
# Configuracion para ocultar mensajes de advertencia de TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Nombres de las clases de CIFAR-10
CLASS_NAMES = [
    "avión",
    "coche",
    "pájaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camión",
]


# =============================================================================
# 2. FUNCIONES DE CARGA Y PREPROCESAMIENTO DE DATOS
# =============================================================================
def cargar_y_preprocesar_cifar10_mlp():
    """
    Carga el dataset CIFAR-10 y lo preprocesa para un MLP.
    - Normaliza las imágenes a [0, 1].
    - Aplana las imágenes a un vector de 3072.
    - Codifica las etiquetas en one-hot.
    """
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

    # Escala las imagenes a [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Aplana las imagenes a un vector de 3072
    num_pixels = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    X_train = X_train.reshape(X_train.shape[0], num_pixels)
    X_test = X_test.reshape(X_test.shape[0], num_pixels)

    Y_train = keras.utils.to_categorical(Y_train, len(CLASS_NAMES))
    Y_test = keras.utils.to_categorical(Y_test, len(CLASS_NAMES))

    return X_train, Y_train, X_test, Y_test


def show_image(image, title):
    """Funcion para mostrar una imagen con su titulo"""

    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.xticks([])  # Ocultar ejes
    plt.yticks([])  # Ocultar ejes
    plt.show()


def show_train_evolution(history, title="Evolución del entrenamiento"):
    """
    Muestra la evolucion de la precision y la perdida durante el entrenamiento.

    Args:
        history: Objecto devuelto por el metodo model.fit de keras.
        title (str): Titulo principal para la figura
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    # Grafica de Accuracy
    ax1.plot(history.history["accuracy"], label="Precision Entrenamiento")
    ax1.plot(history.history["val_accuracy"], label="Precision Validacion")
    ax1.set_title("Precision por Epoca")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Grafica de Loss
    ax2.plot(history.history["loss"], label="Perdida Entrenamiento")
    ax2.plot(history.history["val_loss"], label="Perdida Validacion")
    ax2.set_title("Perdida por Epoca")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.show()


def show_models_comparations(
    models_names, accuracies, times, title="Comparación de modelos"
):
    """
    Crea una grafica de barras para comparar la precision y el tiempo de entrenamiento de varios modelos.

    Args:
        models_names (list): Lista con nombres de los modelos
        accuracies (list): Lista con las precisiones de cada modelo
        times (list): Lista con el tiempo de entrenamiento de cada modelo
        title (str): Titulo para la grafica.
    """

    x = np.arange(len(models_names))  # Posiciones de las etiquetas
    width = 0.35  # Ancho de las barras

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Barras de Precisión
    rects1 = ax1.bar(
        x - width / 2, accuracies, width, label="Precisión (Test)", color="tab:blue"
    )

    ax1.set_ylabel("Tasa de Acierto (Accuracy)", color="tab:blue")
    ax1.set_xlabel("Modelos")
    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_names, rotation=45, ha="right")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    # Eje Y secundario para el Tiempo
    ax2 = ax1.twinx()
    rects2 = ax2.bar(
        x + width / 2, times, width, label="Tiempo (s)", color="tab:orange"
    )
    ax2.set_ylabel("Tiempo de Entrenamiento (s)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def show_confusion_matriz(Y_true, Y_pred, class_names, title="Matriz de Confusión"):
    """
    Calcula y muestra la amatriz de confusion

    Args:
        Y_true: Etiquetas verdaderas
        Y_pred: Predicciones del modelo
        class_names (list): Lista con los nombres de las clases
        title (str): Titulo para la grafica
    """

    # Las predicciones y etiquetas deben ser indices de clase
    y_true_labels = np.argmax(Y_true, axis=1) if Y_true.ndim > 1 else Y_true
    y_pred_labels = np.argmax(Y_pred, axis=1) if Y_pred.ndim > 1 else Y_pred

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    # Mostrar la matriz de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation="vertical")
    ax.set_title(title)
    plt.show()


# =============================================================================
# 4. FUNCIONES DE LAS TAREAS DE LA PRÁCTICA
# =============================================================================
def tarea_toma_de_contacto():
    """
    Código para la sección 2 de la práctica: Cargar, imprimir dimensiones
    y mostrar ejemplos de CIFAR-10.
    """
    print("--- Ejecutando Tarea: Toma de Contacto con CIFAR-10 ---")
    (X_train, Y_train), (_, _) = keras.datasets.cifar10.load_data()
    print("Dimensiones de los datos originales (X_train):", X_train.shape)

    print("\nMostrando 3 imágenes de ejemplo...")
    for i in sample(range(len(X_train)), 3):
        class_index = Y_train[i][0]
        title = f"Imagen X_train[{i}] - Clase: {CLASS_NAMES[class_index]}"
        show_image(X_train[i], title)


# TAREA 1: Definir, entrenar y evaluar un MLP con Keras
def tarea_mlp1(X_train, Y_train, X_test, Y_test):
    """
    Define, entrena y evalia un MLP basico que sigue el enunciado de la tarea 1

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP1 ---")

    # --- Definir arquitectura del modelo
    model = keras.Sequential()

    # Capa de entrada
    model.add(keras.Input(shape=(X_train.shape[1],)))
    # Capa Oculta: Dense, 48 neruonas y activacion sigmoid
    model.add(layers.Dense(48, activation="sigmoid"))
    # Capa de Salida: Dense, 10 neruonas y activacion softmax
    # softmax convierte las salidas en un vector de probabilidades
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

    # --- Compilar Modelo
    model.compile(
        optimizer="adam",  # Optimizador adam
        loss="categorical_crossentropy",  # FUncion de perdida para la clasificacion
        metrics=["accuracy"],  # Metrica de evaluacion
    )

    # --- Resumen del modelo
    print("\n--- Resumen del modelo ---")
    model.summary()

    # --- Entrenar el modelo
    print("\n--- Entrenando el modelo ---")
    start_time = time.time()
    history = model.fit(
        X_train,
        Y_train,
        epochs=10,  # Numero de veces que se recorre el dataset entero
        batch_size=32,  # Numero de lote
        validation_split=0.1,  # 10% de los datos de entrenamiento se usaran para la validacion
    )
    end_time = time.time()
    training_time = end_time - start_time
    print("Tiempo de entrenamiento:", training_time, "segundos")

    # --- Visualizar Evolucion del entrenamiento
    show_train_evolution(history, "Evolución del entrenamiento MLP1")

    # --- Evaluar modelo con el conjunto de Test
    print("\n--- Evaluando el modelo con el conjunto de Test ---")
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print("Perdida en el conjunto de Test:", test_loss)
    print("Precisión en el conjunto de Test:", test_accuracy)


# TAREA 2: Ajustar el valor del parámetro epochs
def tarea_mlp2(X_train, Y_train, X_test, Y_test):
    """
    Analizar el efecto del numero de epocas y utilizar EarlyStopping

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP2 ---")

    # --- Definir arquitectura del modelo, igual que en mlp1
    model = keras.Sequential()
    model.add(keras.Input(shape=(X_train.shape[1],)))
    model.add(layers.Dense(48, activation="sigmoid"))
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # --- Entrenar el modelo con 100 epocas, sin EarlyStopping
    print("\n--- Entrenando el modelo con 100 epocas ---")
    history = model.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
    )

    # --- Configurar el callback EarlyStopping
    print("\n--- Configurando el callback EarlyStopping ---")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",  # monitorea la perdida en el conjunto de validacion
        patience=10,  # esperara 10 epocas sin mejora antes de parar
        restore_best_weights=True,  # asegura quedarse con el mejor modelo
        verbose=1,  # imprime un mensaje cuando para
    )

    # --- Entrenar el modelo con 100 epocas, EarlyStopping decide cuando parar
    print("\n--- Entrenando el modelo con 100 epocas y EarlyStopping ---")
    history = model.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1,
    )

    # -- Visualización y evaluacion
    print(
        "\nEl entrenamiento se ha detenido en la epoca:", early_stopping.stopped_epoch
    )

    show_train_evolution(history, "Evolución del entrenamiento MLP2")

    print("\n--- Evaluando el modelo con el conjunto de Test ---")
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

    print("Perdida en el conjunto de Test:", test_loss)
    print("Precisión en el conjunto de Test:", test_accuracy)


# =============================================================================
# 5. BLOQUE DE EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # --- Tarea: Toma de contacto ---
    # Descomenta la siguiente línea para ejecutar el código de la sección 2
    # tarea_toma_de_contacto()

    # --- Carga de datos para los modelos ---
    print("Cargando y preprocesando datos para MLP...")
    X_train_mlp, Y_train_mlp, X_test_mlp, Y_test_mlp = (
        cargar_y_preprocesar_cifar10_mlp()
    )
    print("Datos para MLP cargados y listos.")

    # Tarea MLP1: Definir, entrenar y evaluar un MLP con Keras
    # tarea_mlp1(X_train_mlp, Y_train_mlp, X_test_mlp, Y_test_mlp)

    # Tarea MLP2: Ajustar el valor del parámetro epochs
    tarea_mlp2(X_train_mlp, Y_train_mlp, X_test_mlp, Y_test_mlp)
