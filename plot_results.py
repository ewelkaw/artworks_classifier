import matplotlib.pyplot as plt


def plot_results(history, ts):
    acc = history["acc"]
    val_acc = history["val_acc"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(epochs, acc, "r-", label="Training Accuracy")
    axes[0].plot(epochs, val_acc, "b--", label="Validation Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].legend(loc="best")

    axes[1].plot(epochs, loss, "r-", label="Training Loss")
    axes[1].plot(epochs, val_loss, "b--", label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].legend(loc="best")

    plt.show()
    plt.imsave(f"results_{ts}.jpg", fig)
