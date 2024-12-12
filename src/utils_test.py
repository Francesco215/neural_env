import torch
import pytest
from utils import two_hot, two_hot_inv

def test_two_hot_basic():
    x = torch.tensor([0.5, 1.5, 2.5])
    vmin = 0.0
    vmax = 3.0
    num_bins = 3
    result = two_hot(x, vmin, vmax, num_bins)
    expected_shape = (3, num_bins)
    assert result.shape == expected_shape

def test_two_hot_clamping():
    x = torch.tensor([-1.0, 4.0])
    vmin = 0.0
    vmax = 3.0
    num_bins = 3
    result = two_hot(x, vmin, vmax, num_bins)
    expected_shape = (2, num_bins)
    assert result.shape == expected_shape
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)

def test_two_hot_bin_assignment():
    x = torch.tensor([0.5, 1.5, 2.5])
    vmin = 0.0
    vmax = 3.0
    num_bins = 30
    result = two_hot(x, vmin, vmax, num_bins)
    assert torch.all(result.sum(dim=-1).round() == 1)

def test_two_hot_inv_inverts_two_hot():
    x = torch.tensor([0.5, 1.5, 2.5])
    vmin = 0.0
    vmax = 3.0
    num_bins = 32
    encoded = two_hot(x, vmin, vmax, num_bins)
    decoded = two_hot_inv(encoded, vmin, vmax, num_bins)
    assert torch.allclose(x, decoded, atol=1e-2)


if __name__ == '__main__':
    pytest.main()
