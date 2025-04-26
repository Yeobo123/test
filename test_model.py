import torch
from train import get_model, FMNISTDataset

# Nếu file code chính của bạn tên là `train.py`, thì đổi dòng trên thành:
# from train import get_model, FMNISTDataset

def test_model_output_shape():
    model, _, _ = get_model()
    dummy_input = torch.randn(1, 28*28).to(model[0].weight.device)
    output = model(dummy_input)
    assert output.shape == (1, 10), "Output phải có shape (1, 10)"

def test_dataset_length_and_shape():
    # Tạo dữ liệu giả
    dummy_x = torch.randn(100, 28, 28)
    dummy_y = torch.randint(0, 10, (100,))
    
    dataset = FMNISTDataset(dummy_x, dummy_y)

    assert len(dataset) == 100, "Dataset phải có 100 phần tử"

    x, y = dataset[0]
    assert x.shape == (28*28,), "Input phải được flatten thành (784,)"
    assert isinstance(y.item(), int), "Label phải là kiểu int"

def test_device_model():
    model, _, _ = get_model()
    for layer in model:
        if hasattr(layer, 'weight'):
            assert layer.weight.device.type in ["cuda", "cpu"], "Model phải nằm trên CPU hoặc GPU"

