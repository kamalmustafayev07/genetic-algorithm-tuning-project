from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(validation_split=True):
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    X_train_full = X_train_full.reshape(-1, 28, 28, 1).astype("float32") / 255
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255

    if not validation_split:
        return (X_train_full, to_categorical(y_train_full, 10)), (X_test, to_categorical(y_test, 10))

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )

    return (
        X_train, to_categorical(y_train, num_classes=10),
        X_val, to_categorical(y_val, num_classes=10),
        X_test, to_categorical(y_test, num_classes=10)
    )
