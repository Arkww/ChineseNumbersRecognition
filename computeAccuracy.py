import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def computeAccuracy(model, data, class_names):
    model = model.eval()
    correct = 0.0
    totalExamples = 0
    totalLabels = np.array([])
    totalPred = np.array([])

    for idx, (features, labels) in enumerate(data):
        with torch.no_grad():
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features) # run the model to get the predictions

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions  # compare predictions with true label

        correct += torch.sum(compare)
        totalExamples += len(compare)

        totalLabels = np.concatenate((totalLabels, labels.cpu().numpy()))
        totalPred = np.concatenate((totalPred, predictions.cpu().numpy()))

    cm = confusion_matrix(totalLabels, totalPred)  # assemble the confusion matrix and print it
    plot_confusion_matrix(cm, class_names)

    return (correct / totalExamples).item()  # return the accuracy
