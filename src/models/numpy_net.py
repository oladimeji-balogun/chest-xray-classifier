import numpy as np 

# the goal is to creating a convolution layer from scratch

def convolve2D(image: np.array, filter: np.array, stride: int) -> np.array: 
    output = []

    for row in range(image[0]):

        for col in range(image):
            # get the sub matrix 
            sub_matrix = image[row + stride : col + stride]
            pass 