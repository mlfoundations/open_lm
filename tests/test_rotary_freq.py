import torch
import pytest
from open_lm.positional_embedding.rotary import RotaryEmbedding  # replace 'your_module' with the actual module name

@pytest.fixture
def create_rotary_embedding():
    def _create_rotary_embedding(dim_model, seq_len, frequency):
        return RotaryEmbedding(dim_model, seq_len, frequency)
    return _create_rotary_embedding

def test_frequency_input(create_rotary_embedding):
    dim_model = 32
    seq_len = 64

    # Create two rotary embeddings with different frequencies
    freq1 = 10000
    freq2 = 20000
    rotary1 = create_rotary_embedding(dim_model, seq_len, freq1)
    rotary2 = create_rotary_embedding(dim_model, seq_len, freq2)

    # Generate some dummy data
    q = torch.randn(1, seq_len, dim_model)
    k = torch.randn(1, seq_len, dim_model)

    # Forward pass with different frequencies
    q1, k1 = rotary1(q, k)
    q2, k2 = rotary2(q, k)

    # Ensure the outputs are different
    assert not torch.allclose(q1, q), "The outputs should not be close"
    assert not torch.allclose(k1, k), "The outputs should not be close"
    assert not torch.allclose(q1, q2), "The outputs for different frequencies should not be close"
    assert not torch.allclose(k1, k2), "The outputs for different frequencies should not be close"

    # load the state dicts
    state_dict1 = torch.load("tests/assets/rotary1_old.pt")
    state_dict2 = torch.load("tests/assets/rotary2_old.pt")

    # Build new rotary embeddings with exchanged frequencies
    rotary1_loaded = create_rotary_embedding(dim_model, seq_len, freq2)
    rotary2_loaded = create_rotary_embedding(dim_model, seq_len, freq1)

    # Forward pass with loaded models
    q1_loaded, k1_loaded = rotary1_loaded(q, k)
    q2_loaded, k2_loaded = rotary2_loaded(q, k)

    # Ensure the outputs are the same
    assert torch.allclose(q1, q2_loaded), "The outputs should be the same for the same fequencies before loading the state dict"
    assert torch.allclose(k2, k1_loaded), "The outputs should be the same for the same fequencies before loading the state dict"

    # Assert old state dict is in the old format
    assert "inv_freq" in state_dict1, "The old state dict should contain the inv_freq buffer"

    # Load the state dicts
    rotary1_loaded.load_state_dict(state_dict1, strict=False)
    rotary2_loaded.load_state_dict(state_dict2, strict=False)

    # Ensure the frequencies are not overwritten
    assert rotary1_loaded.frequency == freq2, "Frequency should not be overwritten by load_state_dict"
    assert rotary2_loaded.frequency == freq1, "Frequency should not be overwritten by load_state_dict"
    
    # Forward pass with loaded models
    q1_loaded, k1_loaded = rotary1_loaded(q, k)
    q2_loaded, k2_loaded = rotary2_loaded(q, k)

    # Ensure the outputs are the same
    assert torch.allclose(q1, q2_loaded), "The outputs should be the same for the same fequencies after loading the state dict"
    assert torch.allclose(k2, k1_loaded), "The outputs should be the same for the same fequencies after loading the state dict"
    
    # Ensure the outputs are still different
    assert not torch.allclose(q1_loaded, q2_loaded), "The outputs for different frequencies should not be close"
    assert not torch.allclose(k1_loaded, k2_loaded), "The outputs for different frequencies should not be close"


if __name__ == "__main__":
    pytest.main([__file__])
