import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1D CNN
x_train_1d = np.random.random((100, 10, 1))
y_train_1d = np.random.randint(2, size=(100, 1))

model_1d = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(10, 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_1d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_1d.fit(x_train_1d, y_train_1d, epochs=10)

plt.plot(model_1d.history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# 2D CNN
x_train_2d = np.random.random((100, 10, 10, 3))
y_train_2d = np.random.randint(2, size=(100, 1))

model_2d = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(10, 10, 3)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_2d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_2d.fit(x_train_2d, y_train_2d, epochs=10)

plt.plot(model_2d.history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# 3D CNN
x_train_3d = np.random.random((100, 10, 10, 10, 3))
y_train_3d = np.random.randint(2, size=(100, 1))

model_3d = tf.keras.Sequential([
    tf.keras.layers.Conv3D(32, 3, activation='relu', input_shape=(10, 10, 10, 3)),
    tf.keras.layers.MaxPooling3D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_3d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_3d.fit(x_train_3d, y_train_3d, epochs=10)

plt.plot(model_3d.history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
