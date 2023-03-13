from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(y_true, y_pred, plot_cm=True):
    """
    Display a confusion matrix and classification report based on true, and predicted values.

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        plot_cm (bool): Whether or not to plot the confusion matrix.

    """
    if plot_cm:
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.close(1) # don't display pre-sized matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax)
        plt.title('Model Confusion Matrix')
        plt.xlabel('Predicted Letter')
        plt.ylabel('True Letter')
        plt.xticks(range(26), labels=range(26))
        plt.yticks(range(26), labels=range(26))
        plt.show()
    # Display classification report
    print(classification_report(y_true, y_pred))
