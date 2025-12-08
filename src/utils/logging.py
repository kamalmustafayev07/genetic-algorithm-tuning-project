def log_params(eval_id, params):
    print(f"\nEvaluation {eval_id}:")
    print(
        "Params: filters1={}, kernel1={}, filters2={}, kernel2={}, dropout={:.3f}, dense1={}, dense2={}, lr={:.6f}".format(
            params["filters1"],
            params["kernel1"],
            params["filters2"],
            params["kernel2"],
            params["dropout_rate"],
            params["dense1"],
            params["dense2"],
            params["learning_rate"]
        )
    )

def log_result(val_acc):
    loss = 1 - val_acc
    print(f"Validation accuracy: {val_acc:.4f} (loss: {loss:.4f})")
    return loss
