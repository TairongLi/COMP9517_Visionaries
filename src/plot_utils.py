"""Plotting helpers for experiment reports."""

import matplotlib.pyplot as plt


def plot_training_history(history_df, title, save_path=None):
    """Plot loss and accuracy curves for train and validation sets."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history_df["epoch"], history_df["train_acc"], marker="o", label="Train accuracy")
    axes[1].plot(history_df["epoch"], history_df["val_acc"], marker="o", label="Validation accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Training curve saved to:", save_path)

    plt.show()
