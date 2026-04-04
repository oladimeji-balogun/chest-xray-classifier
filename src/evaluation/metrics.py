import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    accuracy_score,
    confusion_matrix
)
import numpy as np

from ..models.cnn import ChestXRayCNN

def evaluate(
    model: nn.Module, 
    loader: DataLoader,
    device: torch.device
): 
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad(): 
        for images, labels in loader: 
            images, labels = images.to(device), labels.to(device)
            model = model.to(device)

            output = model(images)
            preds = torch.argmax(output, dim=1)
            probs = torch.softmax(output, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(y_true=all_labels, y_pred=all_preds)
    matrix = confusion_matrix(y_true=all_labels, y_pred=all_preds)
    roc_auc = roc_auc_score(y_true=all_labels, y_score=all_probs)
    clfxn_report = classification_report(y_true=all_labels, y_pred=all_preds)
    evaluation = {
        "accuracy": accuracy, 
        "confusion_matrix": matrix, 
        "roc_auc": roc_auc,
        "clfxn_report": clfxn_report
    }
    return evaluation


def evaluate_at_threshold(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        threshold: float = 0.5
    ) -> dict:
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            probs = torch.softmax(output, dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # apply threshold
    preds = (all_probs >= threshold).astype(int)

    # compute metrics same as before
    accuracy = accuracy_score(y_true=all_labels, y_pred=preds)
    matrix = confusion_matrix(y_true=all_labels, y_pred=preds)
    roc_auc = roc_auc_score(y_true=all_labels, y_score=all_probs)
    clfxn_report = classification_report(y_true=all_labels, y_pred=preds)

    normal_recall = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    pneumonia_recall = matrix[1, 1] / (matrix[1, 0] + matrix[1, 1])

    evaluation = {
        "accuracy": accuracy,
        "confusion_matrix": matrix,
        "roc_auc": roc_auc,
        "clfxn_report": clfxn_report,
        "normal_recall": normal_recall,
        "pneumonia_recall": pneumonia_recall
    }
    return evaluation

    
# if __name__ == "__main__": 
#     model = ChestXRayCNN(n_classes=2, image_size=224).to("cuda")
#     model.load_state_dict(torch.load("best-model.pth"))

#     evaluation = evaluate_at_threshold(model=model)

