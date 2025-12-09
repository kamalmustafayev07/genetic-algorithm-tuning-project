from src.data.data_loader import load_data
from src.models.model_builder import build_model_from_vector
from src.utils.saver import save_json
from src.utils.config import TRAINING
import json
import matplotlib.pyplot as plt
import os

def final_evaluation():
    with open("results/metrics/best_params.json") as f:
        best_vector = json.load(f)["best_vector"] 

    (X_train_full, y_train_cat), (X_test, y_test_cat) = load_data(validation_split=False)

    model, params = build_model_from_vector(best_vector)

    history = model.fit(
        X_train_full, y_train_cat,
        epochs=TRAINING["epochs_final"],
        batch_size=TRAINING["batch_size"],
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    # Save trained model
    os.makedirs("results/models", exist_ok=True)
    model_save_path = "results/models/final_best_model.keras"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("==============================\n")

    save_json("results/metrics/final_test_results.json", {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "best_parameters": params
    })

    plt.figure(figsize=(12, 4))

    final_acc = history.history['accuracy'][-1]
    final_loss = history.history['loss'][-1]

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.title("Accuracy")
    plt.grid(True)
    plt.text(0.5, 0.05, f"Final Acc: {final_acc:.4f}", 
         transform=plt.gca().transAxes, ha="center", fontsize=10)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title("Loss")
    plt.grid(True)
    plt.text(0.5, 0.05, f"Final Loss: {final_loss:.4f}", 
         transform=plt.gca().transAxes, ha="center", fontsize=10)

    # Saving results
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/final_training_history.png")
    plt.close()
    print("Final test completed.")


if __name__ == "__main__":
    final_evaluation()
