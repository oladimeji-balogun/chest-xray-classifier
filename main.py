from src.models.numpy_net import conv2D, max_pool2d, forward
import numpy as np 

image = np.random.randn(8, 8, 3)    # small RGB image
filters = np.random.randn(3, 3, 3, 8)  # 8 filters, each 3x3x3

output = forward(image, filters)
print(output.shape)  # work out what this should be before running it