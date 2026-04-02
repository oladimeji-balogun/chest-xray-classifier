from src.training.trainer import train
import torch 
from src.models.cnn import TinyCNN

class TestTrainer:
    def test_loss_decreases(self):
        x = torch.randn(1, 1, 28, 28)
        labels = torch.tensor([0])
        model = TinyCNN(n_filters=8, n_classes=2)

        loss_history = train(
            model=model, 
            n_epochs=5, 
            lr=0.0001,
            images=x, 
            labels=labels
        )
        print(f"loss history: {loss_history}")
