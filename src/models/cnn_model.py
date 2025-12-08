from src.data.data_loader import load_data
from src.models.model_builder import build_model_from_vector
from src.utils.config import TRAINING
from src.utils.logging import log_params, log_result
import tensorflow as tf

eval_count = 0
X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat = load_data()

def evaluate_solution(vector):
    global eval_count
    eval_count += 1

    try:
        model, params = build_model_from_vector(vector)

        log_params(eval_count, params)

        history = model.fit(
            X_train, y_train_cat,
            epochs=TRAINING["epochs_ga"],
            batch_size=TRAINING["batch_size"],
            verbose=0,
            validation_data=(X_val, y_val_cat)
        )

        val_acc = max(history.history["val_accuracy"])
        loss = log_result(val_acc)

        tf.keras.backend.clear_session()
        del model

        return loss

    except Exception as e:
        print(f"Error in evaluation {eval_count}: {e}")
        return 1.0
