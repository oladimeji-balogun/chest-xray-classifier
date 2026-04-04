
from src.data.dataset import get_train_transform, get_val_transform, ChestXrayDataset, get_dataloaders
from torch.utils.data import random_split
from src.training.trainer import train
from src.models.cnn import TinyCNN, ChestXRayCNN
import torch
from src.evaluation.metrics import evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

full_train = ChestXrayDataset(root_dir="data/raw/chest_xray/train", transform=get_train_transform(img_size=224))
test_dataset = ChestXrayDataset(root_dir="data/raw/chest_xray/test", transform=get_val_transform(img_size=224))

# train_size = int(0.8 * len(full_train))
# val_size = len(full_train) - train_size

# train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

full_train_indices = list(range(len(full_train)))
image_labels = full_train.image_labels

train_indices, val_indices = train_test_split(
    full_train_indices, 
    test_size=0.2, 
    stratify=image_labels, 
    random_state=42
)

train_dataset = Subset(
    dataset=full_train, 
    indices=train_indices
)

val_dataset = Subset(
    dataset=full_train, 
    indices=val_indices
)
# prepare loaders 
train_loader, test_loader, val_loader = get_dataloaders(
    train_dataset=train_dataset, 
    test_dataset=test_dataset, 
    val_dataset=val_dataset
)



# set up the training for the dataloeaders 
# model = TinyCNN(n_filters=8, n_classes=2)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ChestXRayCNN(n_classes=2, image_size=224).to(device)


normal_count = 1341
pneumonia_count = 3875 
total = normal_count + pneumonia_count
normal_weight = total / (2 * normal_count)
pneumonia_weight = total / (2 * pneumonia_count)

weights = torch.tensor([normal_weight, pneumonia_weight], dtype=torch.float32)
print("started training!")
history = train(
    model=model, 
    n_epochs=25, 
    lr=0.0001, 
    device=device, 
    train_loader=train_loader, 
    test_loader=test_loader, 
    val_loader=val_loader, 
    class_weights=weights
)

print(f"the distribution of the validation set")
from collections import Counter
# if using random_split, check label distribution
labels = [train_dataset.dataset.image_labels[i] for i in val_dataset.indices]
print(Counter(labels))

print("training successful")
print("evaluating the best model")
model.load_state_dict(torch.load('best-model.pth'))
evaluation = evaluate(model=model, loader=test_loader, device=device)
print(f"metrics: {evaluation}")
