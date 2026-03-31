import pytest 
from src.models.numpy_net import (
    conv2D, 
    ReLU, 
    max_pool2d, 
    softmax, 
    flatten, 
    cross_entropy_loss, 
    fc_forward, 
    fc_backward,
    relu_backward
)
import numpy as np 


class TestReLU: 
    def test_kills_negatives(self): 
        x = np.array([-34, -35, -36, -37, 80, 90])
        assert np.allclose(ReLU(x), [0, 0, 0, 0, 80, 90])
    
    def test_3d_array(self): 
        x = np.random.randn(3, 3, 8)
        out = ReLU(x)
        assert x.shape == out.shape 
        assert np.all(out >= 0)

    
class TestMaxPool2D:
    def test_output_shape(self): 
        feature_map = np.random.randn(4, 4, 2)
        pooled = max_pool2d(feature_map=feature_map)
        assert pooled.shape == (2, 2, 2)

class TestCrossEntropy: 
    def test_penalty_on_wrongness(self): 
        x = np.array([0.05, 0.95])
        loss = cross_entropy_loss(probs=x, true_class=1)
        assert np.round(loss, 2) == 0.05


class TestSoftmax: 
    def test_summation_to_zero(self): 
        x = np.random.randn(5)
        softmax_x = softmax(x)
        assert np.isclose(np.sum(softmax_x), 1.0)

class TestFCLayer:
    def test_fc_forward_shape(self): 
        W = np.random.randn(4, 8)
        x = np.random.randn(8)
        b = np.random.randn(4)
        output = fc_forward(W=W, x=x, b=b)
        assert output.shape == (4,)

    def test_fc_backward_shapes(self): 
        dout = np.random.randn(4)
        x = np.random.randn(8)
        W = np.random.randn(4, 8)
        b = np.random.randn(4)

        dL_dW, dL_db, dL_dx = fc_backward(dout=dout, x=x, W=W)
        assert dL_dW.shape == W.shape
        assert dL_dx.shape == x.shape
        assert dL_db.shape == b.shape

class TestReLUBackwards: 
    def test_block_negative_gradients(self): 
        x = np.array([1, 2, -3, -4, 5])
        dout = np.array([1, 1, 1, 1, 1])
        gradient = relu_backward(dout=dout, x=x)
        assert np.allclose(gradient, np.array([1, 1, 0, 0, 1]))

    def test_scale_positive_gradients(self): 
        x = np.array([4, 5, -2, 3])
        dout = np.array([3, 3, 3, 3])
        gradient = relu_backward(dout=dout, x=x)
        assert np.allclose(gradient, np.array([3, 3, 0, 3]))