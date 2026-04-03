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

    
