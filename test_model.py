# test_model.py

import torch
from train import get_model  # đổi 'your_main_file_name' thành tên file code chính (bỏ .py)

def test_model_output_shape():
    model, _, _ = get_model()
    dummy_input = torch.randn(1, 28 * 28)
    output = model(dummy_input)
    assert output.shape == (1, 10), "Output shape should be (1, 10)"