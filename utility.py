import matplotlib.pyplot as plt
import numpy as np

def plot_show(H, epochs):
    plt.style.use("ggplot")
    plt.figure()

    # Plot the training loss and accuracy
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["category_output_loss"], label="train_category_loss")
    plt.plot(np.arange(0, N), H.history["color_output_loss"], label="train_color_loss")
    plt.plot(np.arange(0, N), H.history["category_output_accuracy"], label="train_category_acc")
    plt.plot(np.arange(0, N), H.history["color_output_accuracy"], label="train_color_acc")

    # Plot the validation loss and accuracy
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["val_category_output_loss"], label="val_category_loss")
    plt.plot(np.arange(0, N), H.history["val_color_output_loss"], label="val_color_loss")
    plt.plot(np.arange(0, N), H.history["val_category_output_accuracy"], label="val_category_acc")
    plt.plot(np.arange(0, N), H.history["val_color_output_accuracy"], label="val_color_acc")

    # Add title and labels to the plot
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()
