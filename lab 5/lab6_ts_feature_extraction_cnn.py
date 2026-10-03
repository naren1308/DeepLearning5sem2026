#CH.SC.U4AIE24077
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
np.random.seed(42)
def generate_time_series(num_samples, seq_len, class_type):
    X = []
    y = []
    time = np.linspace(0, 10, seq_len)
    for _ in range(num_samples):
        if class_type == 0:
            freq = np.random.uniform(1.0, 2.0)
            signal = np.sin(2 * np.pi * freq * time)
        else:
            freq = np.random.uniform(1.0, 2.0)
            signal = np.sign(np.sin(2 * np.pi * freq * time))
        noise = np.random.normal(0, 0.3, seq_len)
        X.append(signal + noise)
        y.append(class_type)
    return np.array(X), np.array(y)
num_samples_per_class = 500
seq_len = 100
X_class0, y_class0 = generate_time_series(num_samples_per_class, seq_len, 0)
X_class1, y_class1 = generate_time_series(num_samples_per_class, seq_len, 1)
X = np.concatenate([X_class0, X_class1], axis=0)
y = np.concatenate([y_class0, y_class1], axis=0)
X = X.reshape(-1, seq_len, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
model = Sequential([
    Input(shape=(seq_len, 1)),
    Conv1D(filters=32, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
print("Training the 1D CNN model...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()
plt.show()
feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[0].output
)
sample_input = X_test[0:1]
feature_maps = feature_extractor.predict(sample_input)
plt.figure(figsize=(10, 6))
plt.suptitle("First 4 Feature Maps from Conv1D Layer (Sample: Sine/Square Wave)")
for i in range(4):
    plt.subplot(4, 1, i+1)
    plt.plot(feature_maps[0, :, i])
    plt.ylabel(f"Filter {i}")
plt.xlabel("Time Step")
plt.tight_layout()
plt.show()
