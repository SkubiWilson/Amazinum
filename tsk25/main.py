import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Завантаження датасету
def load_dataset():
    with h5py.File('train_signs.h5', "r") as train_dataset:
        X_train = np.array(train_dataset["train_set_x"][:])
        Y_train = np.array(train_dataset["train_set_y"][:])
    with h5py.File('test_signs.h5', "r") as test_dataset:
        X_test = np.array(test_dataset["test_set_x"][:])
        Y_test = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    Y_train = Y_train.reshape((1, Y_train.shape[0]))
    Y_test = Y_test.reshape((1, Y_test.shape[0]))

    return X_train, Y_train, X_test, Y_test, classes

# Показ прикладів зображень
def display_samples_in_grid(X, n_rows, n_cols=None, y=None):
    if n_cols is None:
        n_cols = n_rows
    indices = np.random.randint(0, len(X), n_rows * n_cols)
    for i in range(n_rows):
        for j in range(n_cols):
            index = n_cols * i + j
            ax = plt.subplot(n_rows, n_cols, index + 1)
            ax.imshow(X[indices[index]], cmap='Greys')
            if y is not None:
                plt.title(int(y[indices[index]]))
            plt.axis('off')

# Завантаження даних
train_data, train_labels, test_data, test_labels, classes = load_dataset()

print('train_data.shape =', train_data.shape)
print('train_labels.shape =', train_labels.shape)
print('test_data.shape =', test_data.shape)
print('test_labels.shape =', test_labels.shape)

# Візуалізація зображень
plt.figure(figsize=(12, 8))
display_samples_in_grid(train_data, n_rows=4, n_cols=6, y=train_labels.T)
plt.tight_layout(h_pad=1, w_pad=1)
plt.show()

# Нормалізація та one-hot encoding
X_train = train_data / 255.0
X_test = test_data / 255.0
Y_train_cat = to_categorical(train_labels[0])
Y_test_cat = to_categorical(test_labels[0])

# Побудова моделі
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(128, activation='tanh'),
    Dropout(0.5),
    Dense(64, activation='tanh'),
    Dropout(0.3),
    Dense(6, activation='softmax')
])

# Компіляція
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Навчання
history = model.fit(X_train, Y_train_cat,
                    epochs=15,
                    batch_size=64,
                    validation_data=(X_test, Y_test_cat))

# Візуалізація результатів
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
