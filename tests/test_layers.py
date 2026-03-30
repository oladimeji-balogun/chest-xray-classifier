import pytest 
from src.models.numpy_net import (
    conv2D, 
    ReLU, 
    max_pool2d, 
    softmax, 
    flatten
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