import numpy as np 


def conv2D(image: np.ndarray, filterrs: np.ndarray, stride: int = 1) -> np.ndarray: 

    image_h, image_w, in_channels = image.shape 
    filter_h, filter_w, in_channels, out_channels = filterrs.shape

    output_h = (image_h - filter_h) // stride + 1
    output_w = (image_w - filter_w) // stride + 1

    result = np.zeros((output_h, output_w, out_channels))

    for i in range(output_h): # iterator over the rows
        for j in range(output_w): # iterator over the colums 
            for f in range(out_channels):
                image_row = i * stride 
                image_col = j * stride 

                submatrix = image[image_row:image_row + filter_h, image_col: image_col + filter_w, :]

                convolution_matrix = submatrix * filterrs[:, :, :, f]
                result[i][j][f] = np.sum(convolution_matrix)
            
    return result


def ReLU(x: np.ndarray) -> np.ndarray: 
    return np.maximum(0, x)

def max_pool2d(feature_map: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    # feature_map shape: (H, W, channels)
    # output shape: (output_h, output_w, channels)
    # at each position, take the max value in the pool_size x pool_size window

    feature_h, feature_w, n_channels = feature_map.shape 
    output_h = (feature_h - pool_size) // stride + 1
    output_w = (feature_w - pool_size) // stride + 1

    result = np.zeros((output_h, output_w, n_channels))
    for i in range(output_h): 
        for j in range(output_w): 
            for k in range(n_channels): 
                image_h = i * stride 
                image_w = j * stride

                submatrix = feature_map[image_h:image_h + pool_size, image_w: image_w + pool_size, :]
                result[i][j][k] = np.max(submatrix[:, :, k])
    return result 

def forward(image: np.ndarray, filters: np.ndarray) -> np.ndarray: 
    convolved = conv2D(image=image, filterrs=filters, stride=2)
    activated = ReLU(x=convolved)
    pooled = max_pool2d(feature_map=activated)
    return pooled 

def softmax(x: np.ndarray) -> np.ndarray: 
    
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def flatten(x: np.ndarray) -> np.ndarray: 
    h, w, d = x.shape 
    return x.reshape(h * w * d, )

def cross_entropy_loss(probs: np.ndarray, true_class: int) -> float: 
    p_correct = probs[true_class]
    loss = -1 * np.log(p_correct + 1e-7)
    return loss

def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray: 
    return dout * (x > 0)

def fc_forward(W: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray: 
    output = W @ x
    output += b
    return output 

def fc_backward(dout: np.ndarray, x: np.ndarray, W: np.ndarray):
    # dout: gradient flowing in from ahead
    # x: original input during forward pass
    # W: weight matrix
    # return: dL/dW, dL/db, dL/dx
    # use the formulas you derived on paper
    
    dL_dW = np.outer(dout, x)
    dL_dx = np.transpose(W) @ dout
    dL_db = dout
    return dL_dW, dL_db, dL_dx 