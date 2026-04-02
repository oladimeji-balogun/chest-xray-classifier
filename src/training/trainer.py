from ..models.cnn  import TinyCNN
import torch 
import torch.nn as nn


def train(
        model: TinyCNN, 
        n_epochs: int, 
        lr: float, 
        images: torch.Tensor, 
        labels: torch.Tensor
): 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    
    loss_history = []
    # the training loop 
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
    return loss_history