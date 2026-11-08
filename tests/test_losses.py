import torch

from open_lm.losses import CrossEntropyLossWithZLoss


def test_z_loss_ignores_masked_targets():
    torch.manual_seed(0)
    loss_fn = CrossEntropyLossWithZLoss(eps=1e-2)
    logits = torch.randn(4, 10) * 5
    targets = torch.randint(0, 10, (4,))
    expected = loss_fn(logits[:2], targets[:2])

    masked_targets = targets.clone()
    masked_targets[2:] = -100
    torch.testing.assert_close(loss_fn(logits, masked_targets), expected)
