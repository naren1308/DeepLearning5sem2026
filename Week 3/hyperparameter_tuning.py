#CH.SC.U4AIE24077
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
iris_dataset = load_iris()
feature_data = iris_dataset.data
class_labels = iris_dataset.target
feature_scaler = StandardScaler()
feature_data = feature_scaler.fit_transform(feature_data)
train_features, test_features, train_labels, test_labels = train_test_split(
    feature_data,
    class_labels,
    test_size=0.2,
    random_state=42
)
configurations = [
    (0.001, 16, 0.0),
    (0.001, 32, 0.2),
    (0.01, 32, 0.2),
    (0.01, 64, 0.3),
    (0.01, 64, 0.5)
]
for learning_rate, hidden_units, dropout_rate in configurations:
    network_model = Sequential([
        Input(shape=(4,)),
        Dense(hidden_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')
    ])
    network_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model_history = network_model.fit(
        train_features,
        train_labels,
        validation_data=(test_features, test_labels),
        epochs=50,
        verbose=0
    )
    final_train_loss = model_history.history['loss'][-1]
    final_val_acc = model_history.history['val_accuracy'][-1]
    print(
        f"LR={learning_rate}, Hidden={hidden_units}, "
        f"Dropout={dropout_rate}, "
        f"Train Loss={final_train_loss:.4f}, "
        f"Val Acc={final_val_acc:.4f}"
    )
