import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# Завантаження датасету
def load_dataset():
    with h5py.File('train_signs.h5', "r") as train_dataset:
        X_train = np.array(train_dataset["train_set_x"][:])
        Y_train = np.array(train_dataset["train_set_y"][:])
    with h5py.File('test_signs.h5', "r") as test_dataset:
        X_test = np.array(test_dataset["test_set_x"][:])
        Y_test = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])
    return X_train, Y_train, X_test, Y_test, classes


# Показ прикладів
def display_samples(X, y, rows=4, cols=6):
    indices = np.random.randint(0, len(X), rows * cols)
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X[indices[i]])
        plt.title(str(y[indices[i]]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Завантаження даних
X_train, y_train, X_test, y_test, classes = load_dataset()
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Візуалізація
display_samples(X_train, y_train)

# Нормалізація
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train_cat = to_categorical(y_train, num_classes=6)
y_test_cat = to_categorical(y_test, num_classes=6)

# Побудова CNN-моделі
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

# Навчання
history = model.fit(X_train, y_train_cat,
                    epochs=30,
                    batch_size=32,
                    validation_data=(X_test, y_test_cat),
                    callbacks=callbacks)

# Графіки точності/втрат
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Оцінка
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n Final Test Accuracy: {test_acc:.4f}")

# Прогнози
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Класифікаційний звіт
print("\n Classification Report:")
print(classification_report(y_test, y_pred_classes))


plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
