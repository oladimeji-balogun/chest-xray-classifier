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
    p_correct = np.clip(p_correct, 1e-7, 1.0)
    loss = -np.log(p_correct)
    return loss

def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray: 
    return dout * (x > 0)

def fc_forward(W: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray: 
    output = W @ x
    output += b
    return output 

def max_pool_backward(dout: np.ndarray, original: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    
    orig_h, orig_w, n_channels = original.shape
    out_h, out_w, channels = dout.shape
    
    # output gradient same shape as original
    dx = np.zeros_like(original)
    
    for i in range(out_h):
        for j in range(out_w):
            for k in range(n_channels):
                # 1. find window start position in original
                row = i * stride 
                col = j * stride
                
                # 2. extract the window for this channel
                window = original[row:row+pool_size, col:col+pool_size, k]
                
                # 3. find position of max value in window
                flat_idx = np.argmax(window)
                max_row, max_col = np.unravel_index(flat_idx, (pool_size, pool_size))
                
                # 4. route gradient to that position
                dx[row + max_row, col + max_col, k] += dout[i, j, k]
    
    return dx

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


# creating the simple CNN 
class TinyCNN: 
    def __init__(
            self, 
            filter_size: int, 
            n_filters: int, 
            n_classes: int, 
            img_size: int, 
            conv_stride: int, 
            pool_stride: int, 
            pool_size: int
        ):
        self.filters = np.random.randn(filter_size, filter_size, 1, n_filters)
        
        conv_out_h = (img_size - filter_size) // conv_stride + 1
        conv_out_w = (img_size - filter_size) // conv_stride + 1
        pool_out_h = (conv_out_h - pool_size) // pool_stride + 1
        pool_out_w = (conv_out_w - pool_size) // pool_stride + 1
        fc_input_size = pool_out_h * pool_out_w * n_filters

        self.weights = np.random.randn(n_classes, fc_input_size) * 0.001
        self.bias = np.zeros(n_classes)

    def forward(self, image: np.ndarray) -> tuple: 
        conv = conv2D(image=image, filterrs=self.filters, stride=2)
        relued = ReLU(x=conv)
        pooled = max_pool2d(feature_map=relued, pool_size=2, stride=2)
        flattened = flatten(x=pooled)
        output = fc_forward(W=self.weights, x=flattened, b=self.bias)
        probs = softmax(x=output)

        cache = {
            "relu_out": relued, 
            "flattened":flattened, 
            "fc_out": output, 
            "conv_out": conv, 
            "pooled_out": pooled
        }
        return probs, cache
    
    def backward(self, probs: np.ndarray, true_class: int, cache: dict) -> dict: 
        # compute loss gradient
        # run backward through fc layer
        # run backward through relu
        # return gradients
        one_hot = np.zeros_like(probs)
        one_hot[true_class] = 1 
        dL_dscores = probs - one_hot
        dL_dW, dL_db, dL_dx = fc_backward(dout=dL_dscores, x=cache["flattened"], W=self.weights)

        dL_dpooled = dL_dx.reshape(cache["pooled_out"].shape)

        dL_conv_input = max_pool_backward(
            dout=dL_dpooled, 
            original=cache["relu_out"], 
            pool_size=2, 
            stride=2
        )
        dL_dconv = relu_backward(dout=dL_conv_input, x=cache["conv_out"])

        return {
            "dL_dW": dL_dW, 
            "dL_db": dL_db, 
            "dL_conv": dL_dconv
        }

    def update(self, gradients: dict, lr: float): 
        self.weights = self.weights - lr * gradients["dL_dW"]
        self.bias = self.bias - lr * gradients["dL_db"]
        print(f"dL_dW mean: {np.mean(np.abs(gradients['dL_dW'])):.6f}")

    def train_step(self, image: np.ndarray, true_class: int, lr: float) -> float: 
        probs, cache = self.forward(image=image)
        gradients = self.backward(probs=probs, true_class=true_class, cache=cache)
        self.update(gradients=gradients, lr=lr)
        loss = cross_entropy_loss(probs=probs, true_class=true_class)
        print(f"probs: {probs}")
        print(f"loss: {loss}")
        return loss


if __name__ == "__main__": 
    model = TinyCNN(
        filter_size=3, 
        n_filters=8,
        n_classes=2,
        img_size=28, 
        conv_stride=2, 
        pool_stride=2, 
        pool_size=2
    )


    fake_image = np.random.randn(28, 28, 1)
    true_class = 0

    for i in range(20):
        loss = model.train_step(image=fake_image, true_class=true_class, lr=0.0001)
        print(f"iteration: {i+1} loss: {loss:.4f}")