from ..models.cnn  import TinyCNN
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F 

def train(
        model: TinyCNN, 
        n_epochs: int, 
        lr: float,
        device: torch.device, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        val_loader: DataLoader, 
        class_weights: torch.Tensor = None
): 
    if class_weights is not None: 
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else: 
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    history = {
        "train_loss": [], 
        "val_loss": [],  
        "val_accuracy": []
    }
    
    model = model.to(device)

    for epoch in range(n_epochs):

        # funning over the epochs
        model.train()
        train_loss = 0.0

        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)
            # now let's train
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

            # accumulate the loss
            train_loss += loss.item()

        average_train_loss = train_loss / len(train_loader)

        # now let's evaluate 
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0   

        with torch.no_grad():
            for images, labels in val_loader: 
                images, labels = images.to(device), labels.to(device)

                output = model(images) 
                loss = criterion(output, labels)
                val_loss += loss.item()

                probs = F.softmax(input=output, dim=1)
                preds = torch.argmax(input=probs, dim=1)

                correct += (preds == labels).sum().item()
                total += len(images)

        average_val_loss = val_loss / len(val_loader)
        val_acc = correct / total


        history["train_loss"].append(average_train_loss)
        history["val_accuracy"].append(val_acc)
        history["val_loss"].append(average_val_loss)

        # print the resulst

        print(
            f"epoch: {epoch + 1}/{n_epochs} | "
            f"train_loss: {average_train_loss:.4f} | "
            f"val_loss: {average_val_loss:.4f} | "
            f"val_accuracy: {val_acc:.4f}"
        )

    return history
