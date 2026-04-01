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
    