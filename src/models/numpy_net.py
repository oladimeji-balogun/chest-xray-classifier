import numpy as np 

# the goal is to creating a convolution layer from scratch



def conv2D(image: np.ndarray, filterr: np.ndarray, stride: int = 1) -> np.ndarray: 

    image_h, image_w = image.shape 
    filter_h, filter_w = filterr.shape

    output_h = (image_h - filter_h) // stride + 1
    output_w = (image_w - filter_w) // stride + 1

    result = np.zeros((output_h, output_w))

    for i in range(output_h): # iterator over the rows
        for j in range(output_w): # iterator over the colums 

            image_row = i * stride 
            image_col = j * stride 

            submatrix = image[image_row:image_row + filter_h, image_col: image_col + filter_w]

            convolution_matrix = submatrix * filterr 
            result[i][j] = np.sum(convolution_matrix)

    return result
