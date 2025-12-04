from random import sample
from tensorflow import keras
from keras import layers, callbacks
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

# Funciones auxiliares para visualizar los resultados de los modelos entrenados
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time  # Para medir el tiempo de entrenamiento

# =============================================================================
# 1. CONFIGURACION GLOBAL
# =============================================================================
# Configuracion para ocultar mensajes de advertencia de TensorFlow
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Nombres de las clases de CIFAR-10
CLASS_NAMES = [
    "avion",
    "coche",
    "pajaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camion",
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


def show_avg_evolution(histories, title="Evolución Promediada del Entrenamiento"):
    """
    Muestra la evolución promediada de la precisión y la pérdida de varias
    ejecuciones, incluyendo la desviación estándar.

    Args:
        histories (list): Lista de objetos 'history.history' de Keras.
        title (str): Título principal para la figura.
    """
    # Extraer métricas de todas las histories
    accs = np.array([h["accuracy"] for h in histories])
    val_accs = np.array([h["val_accuracy"] for h in histories])
    losses = np.array([h["loss"] for h in histories])
    val_losses = np.array([h["val_loss"] for h in histories])

    # Calcular promedio y desviación estándar a lo largo de las ejecuciones (axis=0)
    mean_acc, std_acc = np.mean(accs, axis=0), np.std(accs, axis=0)
    mean_val_acc, std_val_acc = np.mean(val_accs, axis=0), np.std(val_accs, axis=0)
    mean_loss, std_loss = np.mean(losses, axis=0), np.std(losses, axis=0)
    mean_val_loss, std_val_loss = np.mean(val_losses, axis=0), np.std(
        val_losses, axis=0
    )

    epochs = range(1, len(mean_acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    fig.suptitle(title, fontsize=16)

    # Gráfica de Precisión (Accuracy)
    ax1.plot(epochs, mean_acc, "b", label="Precisión Entrenamiento (Promedio)")
    ax1.fill_between(
        epochs, mean_acc - std_acc, mean_acc + std_acc, color="b", alpha=0.2
    )
    ax1.plot(epochs, mean_val_acc, "r", label="Precisión Validación (Promedio)")
    ax1.fill_between(
        epochs,
        mean_val_acc - std_val_acc,
        mean_val_acc + std_val_acc,
        color="r",
        alpha=0.2,
    )
    ax1.set_title("Precisión Promedio por Época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Precisión")
    ax1.legend()

    # Gráfica de Pérdida (Loss)
    ax2.plot(epochs, mean_loss, "b", label="Pérdida Entrenamiento (Promedio)")
    ax2.fill_between(
        epochs, mean_loss - std_loss, mean_loss + std_loss, color="b", alpha=0.2
    )
    ax2.plot(epochs, mean_val_loss, "r", label="Pérdida Validación (Promedio)")
    ax2.fill_between(
        epochs,
        mean_val_loss - std_val_loss,
        mean_val_loss + std_val_loss,
        color="r",
        alpha=0.2,
    )
    ax2.set_title("Pérdida Promedio por Época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Pérdida")
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
def tarea_test():
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


# ==========
# Parte 1
# ==========

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
    model.add(keras.Input(shape=X_train[0].shape))
    # Aplanar
    model.add(layers.Flatten())
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

# TAREA 2_1: Ajustar el valor del parámetro epochs
def tarea_mlp2_ajuste_manual(
    X_train, Y_train, X_test, Y_test, n_repeticiones=5, epochs=80
):
    """
    Realiza N ejecuciones con un número fijo de épocas para
    analizar el comportamiento promedio del modelo y detectar visualmente el
    sobreentrenamiento.

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP2 ---")

    histories = []
    test_losses = []
    test_accuracies = []

    for i in range(n_repeticiones):
        print(f"\n--- Ejecutando repetición {i+1}/{n_repeticiones} ---")

        # --- Definir arquitectura del modelo, igual que en mlp1
        model = keras.Sequential()
        model.add(keras.Input(shape=X_train[0].shape))
        model.add(layers.Flatten())
        model.add(layers.Dense(48, activation="sigmoid"))
        model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # --- Entrenar el modelo con 80 epocas, sin EarlyStopping
        print("\n--- Entrenando el modelo con 100 epocas ---")
        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
        )

        print(f"Evaluando modelo de la repetición {i+1}...")
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
        test_losses.append(loss)
        test_accuracies.append(acc)

        # 3. Guardar el historial
        histories.append(history.history)
        print(f"Repetición {i+1} finalizada. Test Accuracy: {acc*100:.2f}%")

    # 4. Visualizar los resultados promediados del entrenamiento
    print("\nGenerando gráfica promediada de las ejecuciones...")
    show_avg_evolution(histories, "Evolución Promedio del Entrenamiento (100 Épocas)")

    # --- Calcular y mostrar los resultados promediados de la evaluación ---
    mean_test_acc = np.mean(test_accuracies)
    std_test_acc = np.std(test_accuracies)
    mean_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)

    print("\n--- Resultados Finales Promediados en el Conjunto de Test ---")
    print(
        f"Precisión (Accuracy) Promedio: {mean_test_acc*100:.2f}% (± {std_test_acc*100:.2f}%)"
    )
    print(f"Pérdida (Loss) Promedio:     {mean_test_loss:.4f} (± {std_test_loss:.4f})")
    print("=" * 60)

# TAREA 2_": Ajustar el valor de epochs usando callback EarlyStopping
def tarea_mlp2_early_stopping(X_train, Y_train, X_test, Y_test):
    """
    Comparativa de configuraciones de EarlyStopping.
    Entrena varios modelos variando 'patience' y 'min_delta' para analizar
    el impacto en tiempo y precision.

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """

    print("\n--- Ejecutando Tarea MLP2 - EarlyStopping ---")

    configs = [
        {"patience": 5, "min_delta": 0.01},
        {"patience": 10, "min_delta": 0},
        {"patience": 15, "min_delta": 0},
    ]

    model_names = []
    accuracy_results = []
    time_results = []

    for conf in configs:
        print(f"\n--- Probando Configuracion: {conf} ---")


        model = keras.Sequential()
        model.add(keras.Input(shape=X_train[0].shape))
        model.add(layers.Flatten())
        model.add(layers.Dense(48, activation="sigmoid"))
        model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=conf["patience"],
            min_delta=conf["min_delta"],
            restore_best_weights=True,
            verbose=1,
        )

        start_time = time.time()
        model.fit(
            X_train,
            Y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0,
        )
        end_time = time.time()
        training_time = end_time - start_time

        _, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

        model_names.append(
            f"MLP (patience={conf['patience']}, min_delta={conf['min_delta']})"
        )
        accuracy_results.append(test_accuracy)
        time_results.append(training_time)

        print(
            f"Resultado: Accuracy = {test_accuracy*100:.2f}%, Tiempo = {training_time:.2f}s"
        )

    show_models_comparations(
        model_names,
        accuracy_results,
        time_results,
        "Comparacion de modelos por EarlyStopping con diferente configuracion",
    )

# TAREA 3: Ajustar el valor de 'batch_size'
def tarea_mlp3(X_train, Y_train, X_test, Y_test):
    """
    Analizar el efecto de la cantidad definida del batch_size

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP3 ---")

    # Valores de batch_size a probar
    batch_sizes = [32, 64, 128, 256, 512]

    # Listas para guardar los resultados
    accuracy_results = []
    time_results = []
    model_names = []

    for bs in batch_sizes:
        print("\nEntrenamiento con batch_size =", bs)

        model = keras.Sequential()
        model.add(keras.Input(shape=X_train[0].shape))
        model.add(layers.Flatten())
        model.add(layers.Dense(48, activation="sigmoid"))
        model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )

        start_time = time.time()
        model.fit(
            X_train,
            Y_train,
            epochs=100,
            batch_size=bs,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0,
        )
        end_time = time.time()
        training_time = end_time - start_time

        _, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

        accuracy_results.append(test_accuracy)
        time_results.append(training_time)
        model_names.append(f"MLP (batch_size={bs})")

        print(
            f"Resultado: Accuracy = {test_accuracy*100:.2f}%, Tiempo = {training_time:.2f}s"
        )

    # Grafica comparativa
    show_models_comparations(
        model_names,
        accuracy_results,
        time_results,
        "Comparacion de modelos por batch_size",
    )

# TAREA 4: Probar diferentes funciones de activación
def tarea_mlp4(X_train, Y_train, X_test, Y_test):
    """
    Compara el rendimiento de diferentes combinaciones de funciones de
    activacion y inicializadores de pesos.

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP4 ---")

    # Configuraciones para comparar
    configs = [
        {
            "activation": "sigmoid",
            "initializer": "glorot_uniform",
            "name": "Sigmoid con Glorot",
        },
        {"activation": "relu", "initializer": "he_normal", "name": "ReLU con He"},
        {
            "activation": "leaky_relu",
            "initializer": "he_normal",
            "name": "Leaky ReLU con He",
        },
    ]

    # Listas para guardar los resultados
    accuracy_results = []
    time_results = []
    model_names = []

    for conf in configs:
        print("\nEntrenamiento con configuracion:", conf["name"])

        model = keras.Sequential()
        model.add(layers.Input(shape=X_train[0].shape))
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                48,
                activation=conf["activation"],
                kernel_initializer=conf["initializer"],
            )
        )
        model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )

        start_time = time.time()
        model.fit(
            X_train,
            Y_train,
            epochs=100,
            batch_size=512,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0,
        )
        end_time = time.time()
        training_time = end_time - start_time

        _, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

        accuracy_results.append(test_accuracy)
        time_results.append(training_time)
        model_names.append(conf["name"])

        print(
            f"Resultado: Accuracy = {test_accuracy*100:.2f}%, Tiempo = {training_time:.2f}s"
        )

    # Grafica comparativa
    print("\nGrafica comparativa")
    show_models_comparations(
        model_names,
        accuracy_results,
        time_results,
        "Comparacion de modelos por Funcion de Activacion",
    )

# TAREA 5: Ajustar el numero de neuronas
def tarea_mlp5(X_train, Y_train, X_test, Y_test):
    """
    Analiza el efecto del numero de neuronas en la capa oculta
    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP5 ---")

    accuracy_results = []
    time_results = []
    model_names = []

    # Neuronas para ajustar
    neurons = [32, 64, 128, 256, 512]

    for neuron in neurons:
        print("\nEntrenamiento con", neuron, "neuronas")

        model = keras.Sequential()
        model.add(layers.Input(shape=X_train[0].shape))
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                neuron, activation="leaky_relu", kernel_initializer="he_normal"
            )
        )
        model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )

        start_time = time.time()
        model.fit(
            X_train,
            Y_train,
            epochs=100,
            batch_size=512,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0,
        )
        end_time = time.time()
        training_time = end_time - start_time

        _, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

        accuracy_results.append(test_accuracy)
        time_results.append(training_time)
        model_names.append(f"{neuron} neuronas")

        print(
            f"Resultado: Accuracy = {test_accuracy*100:.2f}%, Tiempo = {training_time:.2f}s"
        )

    # Grafica comparativa
    print("\nGrafica comparativa")
    show_models_comparations(
        model_names,
        accuracy_results,
        time_results,
        "Comparacion de modelos por numero de neuronas",
    )

# TAREA 6: Ajustar el numero de capas y de neuronas por capa
def tarea_mlp6(X_train, Y_train, X_test, Y_test):
    """
    Compara el rendimiento de diferentes arquitecturas de red,
    variando entre el numero de capas y la distribucion de 128 neuronas

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP6 ---")

    accuracy_results = []
    time_results = []
    model_names = []

    architectures = [
        {"layers": [128], "name": "1 capa de 128 neuronas"},
        {"layers": [64, 64], "name": "2 capas de 64 neuronas"},
        {"layers": [96, 32], "name": "2 capas de 96 y 32 neuronas"},
        {"layers": [64, 32, 32], "name": "3 capas de 64, 32 y 32 neuronas"},
    ]

    for arch in architectures:
        print("\nEntrenamiento con arquitectura:", arch["name"])

        model = keras.Sequential()
        model.add(keras.Input(shape=X_train[0].shape))
        model.add(layers.Flatten())

        for neurons in arch["layers"]:
            model.add(
                layers.Dense(
                    neurons, activation="leaky_relu", kernel_initializer="he_normal"
                )
            )

        model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )

        start_time = time.time()
        model.fit(
            X_train,
            Y_train,
            epochs=100,
            batch_size=512,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0,
        )
        end_time = time.time()
        training_time = end_time - start_time

        _, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

        accuracy_results.append(test_accuracy)
        time_results.append(training_time)
        model_names.append(arch["name"])

        print(
            f"Resultado: Accuracy = {test_accuracy*100:.2f}%, Tiempo = {training_time:.2f}s"
        )

    # Grafica comparativa
    show_models_comparations(
        model_names,
        accuracy_results,
        time_results,
        "Comparacion de modelos por arquitectura",
    )

# Tarea 7: Aplicar Batch Normalization
def tarea_mlp7_batch_normalization(X_train, Y_train, X_test, Y_test):
    """
    Aplica Batch Normalization

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP7 ---")

    model = keras.Sequential()
    model.add(keras.Input(shape=X_train[0].shape))
    model.add(layers.Flatten())

    # Capa oculta 1, sin activacion
    # model.add(layers.Dense(96, kernel_initializer="he_normal"))
    # Normalizar
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(96, activation="leaky_relu", kernel_initializer="he_normal"))

    # Actua como capa separada
    # model.add(layers.Activation("leaky_relu"))

    # Capa Oculta 2
    # model.add(layers.Dense(32, kernel_initializer="he_normal"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation="leaky_relu", kernel_initializer="he_normal"))
    # model.add(layers.Activation("leaky_relu"))

    # Capa de Salida
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    print("\nEntrenando modelo con Batch Normalization...")
    start_time = time.time()

    history = model.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=512,  # Mantenemos el batch_size óptimo
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )

    end_time = time.time()
    training_time = end_time - start_time

    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")

    loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

    print(f"\n--- Resultado Batch Normalization ---")
    print(f"Precision: {test_accuracy*100:.2f}%")
    print(f"Perdida: {loss:.4f}")
    print(f"Tiempo: {training_time:.2f}s")
    print(f"Epocas: {len(history.history['loss'])}")

    show_train_evolution(history, "Evolucion del entrenamiento con Batch Normalization")

### Tarea 7: Añade regularizacion con Dropout
def tarea_mlp7_dropout(X_train, Y_train, X_test, Y_test):
    """
    Añade regularización con Dropout a la arquitectura con BN

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP7 ---")

    model = keras.Sequential()
    model.add(keras.Input(shape=X_train[0].shape))
    model.add(layers.Flatten())

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(96, activation="leaky_relu", kernel_initializer="he_normal"))
    # Aplica un dropout del 20%
    model.add(layers.Dropout(0.2))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation="leaky_relu", kernel_initializer="he_normal"))
    model.add(layers.Dropout(0.2))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nEntrenando modelo con Dropout...")
    start_time = time.time()

    history = model.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=512,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0,
    )

    end_time = time.time()
    training_time = end_time - start_time

    print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")

    loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)

    print(f"\n--- Resultado BN + Dropout ---")
    print(f"Precision: {test_accuracy*100:.2f}%")
    print(f"Perdida: {loss:.4f}")
    print(f"Tiempo: {training_time:.2f}s")
    print(f"Epocas: {len(history.history['loss'])}")

    show_train_evolution(history, "Evolucion del entrenamiento con BN + Dropout")

### Tarea 7: Añade Data augmentation
def tarea_mlp7_data_augmentation(X_train, Y_train, X_test, Y_test):
    """
    Añade capas de preprocesamiento para generar variaciones de las imágenes
    durante el entrenamiento y mejorar la generalización

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP7 ---")

    # Definir data_augmentation
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = keras.Sequential()
    model.add(keras.Input(shape=X_train[0].shape))

    # Transforma las imagenes antes de que sea un vector
    model.add(data_augmentation)

    model.add(layers.Flatten())

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(96, activation="leaky_relu", kernel_initializer="he_normal"))
    model.add(layers.Dropout(0.2))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation="leaky_relu", kernel_initializer="he_normal"))
    model.add(layers.Dropout(0.2))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nEntrenando modelo con Data Augmentation...")
    start_time = time.time()
    history = model.fit(
        X_train,
        Y_train,
        epochs=200,
        batch_size=512,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1,
    )
    end_time = time.time()
    training_time = end_time - start_time

    loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)

    print(f"\n--- Resultado Data Augmentation ---")
    print(f"Precision: {test_accuracy*100:.2f}%")
    print(f"Perdida: {loss:.4f}")
    print(f"Tiempo: {training_time:.2f}s")
    print(f"Epocas: {len(history.history['loss'])}")

    show_train_evolution(history, "Evolucion del entrenamiento con Data Augmentation")

### Tarea 7: Callback para el Learning Rate
def tarea_mlp7_learning_rate(X_train, Y_train, X_test, Y_test):
    """
    Añade un Learning Rate Scheduler,
    para controlar la convergencia en las etapas finales.

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP7 ---")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = keras.Sequential()
    model.add(keras.Input(shape=X_train[0].shape))
    model.add(data_augmentation)
    model.add(layers.Flatten())

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(96, activation="leaky_relu", kernel_initializer="he_normal"))
    model.add(layers.Dropout(0.2))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(32, activation="leaky_relu", kernel_initializer="he_normal"))
    model.add(layers.Dropout(0.2))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    # Si val_loss no mejora en 5 epocas, divide el LR entre 20
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1
    )

    print("\nEntrenando modelo con Learning Rate...")
    start_time = time.time()
    history = model.fit(
        X_train,
        Y_train,
        epochs=100,
        batch_size=512,
        validation_split=0.1,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1,
    )
    end_time = time.time()
    training_time = end_time - start_time

    loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)

    print(f"\n--- Resultado Learning Rate ---")
    print(f"Precision: {test_accuracy*100:.2f}%")
    print(f"Perdida: {loss:.4f}")
    print(f"Tiempo: {training_time:.2f}s")
    print(f"Epocas: {len(history.history['loss'])}")

    show_train_evolution(history, "Evolucion del entrenamiento con Learning Rate")

### Tarea 7: Modelo definitivo
def tarea_mlp7_max(X_train, Y_train, X_test, Y_test):
    """
    MLP definitivo maximizado en rendimiento

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: MLP7 PRO MAX ---")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomContrast(0.1),
        ]
    )

    model = keras.Sequential()
    model.add(keras.Input(shape=X_train[0].shape))
    model.add(data_augmentation)
    model.add(layers.Flatten())

    architecture = [512, 256, 128]

    for i, neurons in enumerate(architecture):
        model.add(layers.Dense(neurons, kernel_initializer="he_normal"))

        # Entrena 6 capas sin explotar aun
        model.add(layers.BatchNormalization())

        # Activacion
        model.add(layers.Activation("leaky_relu"))

        # Dropout de mas a menos, porque al principio ka capa es mas grande
        drop_rate = 0.25 if i < 2 else 0.15
        model.add(layers.Dropout(drop_rate))

    # Capa de salida
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,  # Reducción más suave (0.2 en vez de 0.1)
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    start_time = time.time()
    history = model.fit(
        X_train,
        Y_train,
        epochs=200,
        batch_size=512,
        validation_split=0.1,
        callbacks=[
            early_stopping,
            lr_scheduler,
        ],
        verbose=1,
    )
    end_time = time.time()
    training_time = end_time - start_time

    loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)

    print(f"\n--- Resultado MLP7 PRO MAX ---")
    print(f"Precision: {test_accuracy*100:.2f}%")
    print(f"Perdida: {loss:.4f}")
    print(f"Tiempo: {training_time:.2f}s")
    print(f"Epocas: {len(history.history['loss'])}")

    # Matriz de Confusión
    print("\nGenerando Matriz de Confusión...")
    Y_pred = model.predict(X_test)
    show_confusion_matriz(
        Y_test, Y_pred, CLASS_NAMES, "Matriz de Confusión - Modelo Final"
    )

    show_train_evolution(history, "Evolucion del entrenamiento con MLP7 PRO MAX")

# ==========
# Parte 2
# ==========

def tarea_cnn1(X_train, Y_train, X_test, Y_test):
    """
    Definir modelo con CNN y evaluar las epochs

    Args:
        X_train: Datos de entrenamiento
        Y_train: Etiquetas de entrenamiento
        X_test: Datos de test
        Y_test: Etiquetas de test
    """
    print("--- Ejecutando Tarea: CNN1 ---")

    model = keras.Sequential()

    # Capa de entrada CNN
    model.add(layers.Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", input_shape=X_train[0].shape))
    model.add(layers.MaxPooling2D((2,2)))

    # Capa oculta CNN
    model.add(layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal"))
    model.add(layers.MaxPooling2D((2,2)))

    # Aplana
    model.add(layers.Flatten())

    # Capa oculta
    model.add(layers.Dense(100, activation="relu", kernel_initializer="he_normal"))
    
    # Capa de salida
    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    start_time = time.time()
    history = model.fit(
        X_train,
        Y_train,
        epochs=200,
        batch_size=32, # por defecto
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1,
    )
    end_time = time.time()
    training_time = end_time - start_time

    loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)

    print(f"\n--- Resultado CNN1 ---")
    print(f"Precision: {test_accuracy*100:.2f}%")
    print(f"Perdida: {loss:.4f}")
    print(f"Tiempo: {training_time:.2f}s")
    print(f"Epocas: {len(history.history['loss'])}")

    show_train_evolution(history, "Evolucion del entrenamiento con CNN1")



# =============================================================================
# 5. BLOQUE DE EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # tarea_test()

    # --- Carga de datos para los modelos ---
    print("Cargando y preprocesando datos")
    X_train, Y_train, X_test, Y_test = (
        cargar_y_preprocesar_cifar10_mlp()
    )

    #### PARTE 1: MLP
    ### Tarea MLP1: Definir, entrenar y evaluar un MLP con Keras
    # tarea_mlp1(X_train, Y_train, X_test, Y_test)

    ### Tarea MLP2: Ajustar el valor del parámetro epochs
    # Ajuste manual de epocas
    # tarea_mlp2_ajuste_manual(X_train, Y_train, X_test, Y_test)
    # EarlyStopping
    #tarea_mlp2_early_stopping(X_train, Y_train, X_test, Y_test)

    ### Tarea MLP3: Ajustar el valor de 'batch_size'
    # tarea_mlp3(X_train, Y_train, X_test, Y_test)

    ### Tarea MLP4: Probar diferentes funciones de activación
    # tarea_mlp4(X_train, Y_train, X_test, Y_test)

    ### Tarea MLP5: Ajustar el numero de neuronas
    # tarea_mlp5(X_train, Y_train, X_test, Y_test)

    ### Tarea MLP6: Ajustar el numero de capas y de neuronas por capa
    # tarea_mlp6(X_train, Y_train, X_test, Y_test)

    ### Tarea MLP7_BN: Aplicar Batch Normalization
    # tarea_mlp7_batch_normalization(X_train, Y_train, X_test, Y_test)

    ### Tarea MLP7_BN_DO: Regularizar con Dropout
    # tarea_mlp7_dropout(X_train, Y_train, X_test, Y_test)

    ### Tarea MLP7_BN_DO_DA: Aumento de datos
    # tarea_mlp7_data_augmentation(X_train, Y_train, X_test, Y_test)

    ### Tarea MLP7_BN_DO_DA_LR: Callback para el Learning Rate
    # tarea_mlp7_learning_rate(X_train, Y_train, X_test, Y_test)

    ### Tarea MLP7_PRO_MAX
    #tarea_mlp7_max(X_train, Y_train, X_test, Y_test)


    #### PARTE 2: CNN
    ### Tarea CNN1: Definir, entrenar y evaluar un CNN sencillo con Keras
    tarea_cnn1(X_train, Y_train, X_test, Y_test)
