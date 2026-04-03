from src.models.cnn import TinyCNN, ChestXRayCNN
import torch 

class TestTinyCNN:

    def test_output_shape(self): 
        x = torch.randn((1, 1, 28, 28))
        model = TinyCNN(n_filters=8, n_classes=2)
        output = model(x)
        assert output.shape == (1, 2)

    def test_batch_processing(self): 
        x = torch.randn(4, 1, 28, 28)
        model = TinyCNN(n_filters=8, n_classes=2)
        output = model(x)
        assert output.shape == (4, 2)

    def test_output_is_probabilities(self): 
        x = torch.randn(1, 1, 28, 28)
        model = TinyCNN(n_classes=10, n_filters=8)
        output = model(x)

        assert torch.sum(output, dim=1) != 1.0

        import torch.nn.functional as F 
        probs = F.softmax(output, dim=1)
        assert torch.allclose(torch.sum(probs, dim=1), torch.ones(1))

class TestChestXRayCNN: 
    def test_output_shape(self): 
        x = torch.randn(1, 1, 224, 224)
        model = ChestXRayCNN(n_classes=2, image_size=224)
        output = model(x)
        assert output.shape == (1, 2)
