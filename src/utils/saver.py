import os
import json
import matplotlib.pyplot as plt

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def plot_history(values, title, path, ylabel="Loss"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(values, marker="o")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(path)
    plt.close()
