import tensorflow as tf

# Esto debe devolver una lista con tu GPU, no una lista vacía
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"\n🚀 ÉXITO: Tu Mac M2 está listo. Dispositivo detectado: {gpus[0]}")
    print("El entrenamiento usará la aceleración Metal (GPU).")
else:
    print("\n⚠️ ALERTA: No se detecta GPU. Algo falló en la instalación.")
