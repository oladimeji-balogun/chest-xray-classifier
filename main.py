from src.models.numpy_net import conv2D 
import numpy as np 


image = np.array([
    [10, 50, 90, 130], 
    [20, 60, 100, 140],
    [30, 70, 110, 150],
    [40, 80, 120, 160]
])

filterr = np.array([
    [1, 0], 
    [0, -1]
])

conv = conv2D(image=image, filterr=filterr, stride=1)

print(f"the convolution produced: {conv}")