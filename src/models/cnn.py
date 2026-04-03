import torch 
import torch.nn as nn 


class TinyCNN(nn.Module): 
    def __init__(self, n_filters: int, n_classes: int): 
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=n_filters, 
            kernel_size=3, 
            stride=2
        )
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # calculation the flattened shape 
        # after the convolution: 
        conv_output_size = (28 - 3) // 2 + 1 
        # conv shape = (13, 13, 8)

        pool_output_shape = (13 - 2) // 2 + 1
        # pool shape = (6, 6, 8)

        flattened_shape = pool_output_shape * pool_output_shape * n_filters

        self.linear = nn.Linear(in_features=flattened_shape, out_features=n_classes)
        self.relu = nn.ReLU()

        
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x 
    
    

class ChestXRayCNN(nn.Module):
    def __init__(
        self, 
        n_classes: int, 
        image_size: int
    ): 
        super().__init__()

        size_after_conv1 = (image_size - 3) // 2 + 1
        size_after_pool1 = (size_after_conv1 - 2) // 2 + 1

        size_after_conv2 = (size_after_pool1 - 3) // 2 + 1 
        size_after_pool2 = (size_after_conv2 - 2) // 2 + 1

        size_after_conv3 = (size_after_pool2 - 3) // 1 + 1
        size_after_pool3 = (size_after_conv3 - 2) // 2 + 1

        flattened_shape = size_after_pool3 * size_after_pool3 * 128 

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2), 
            nn.BatchNorm2d(num_features=32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1), 
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(),
            nn.Linear(in_features=flattened_shape, out_features=n_classes)
        )





    def forward(self, x: torch.Tensor): 
        x = self.block1(x)
        x = self.block2(x) 
        x = self.block3(x)
        return self.classifier(x)