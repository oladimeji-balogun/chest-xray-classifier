# import torch 
# from src.models.cnn import TinyCNN
# from src.training import trainer

# x = torch.randn(1, 1, 28, 28)
# labels = torch.tensor([0])
# model = TinyCNN(n_filters=8, n_classes=2)

# loss_history = trainer.train(
#     model=model, 
#     n_epochs=20, 
#     lr=0.01,
#     images=x, 
#     labels=labels
# )
# print(f"loss history: {loss_history}")


from src.data.dataset import get_transform, ChestXrayDataset, get_dataloaders
from torch.utils.data import random_split

full_train = ChestXrayDataset(root_dir="data/raw/chest_xray/train", transform=get_transform())
test_dataset = ChestXrayDataset(root_dir="data/raw/chest_xray/test", transform=get_transform())

train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size

train_dataset, val_dataset = random_split(full_train, [train_size, val_size])


# prepare loaders 
train_loader, test_loader, val_loader = get_dataloaders(
    train_dataset=train_dataset, 
    test_dataset=test_dataset, 
    val_dataset=val_dataset
)

# print("the dataloaders are ready!!")
# print(f"len of training set: {len(train_dataset)}")
# print(f"len of test set: {len(test_dataset)}")
# print(f"len of validation set: {len(val_dataset)}")

train_imgs, train_labels = next(iter(train_loader))
print(f"the batch images shape: {train_imgs.shape}")
print(f"the batch labels shape: {train_labels.shape}")
print(f"labels: {train_labels}")


