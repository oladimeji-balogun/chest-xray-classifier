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
        super.__init__()

        # let's start building the network 
        self.relu = nn.ReLU()

        # the first chain
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=32)
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # the second chain
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=64)
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # the third chain
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.batch_norm_3 = nn.BatchNorm2d(num_features=128)
        self.max_pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten(start_dim=1)


    def forward(self, x: torch.Tensor): 
        pass 