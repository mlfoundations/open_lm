from argparse import Namespace
from math import ceil

import torch
import unittest
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torch import nn

from open_lm.train import train_one_epoch


# Dummy model
class SimpleModel(torch.nn.Module):
    def __init__(self, vocab_size, dim=3):
        super(SimpleModel, self).__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.fc = torch.nn.Linear(dim, vocab_size)

    def forward(self, x):
        out = self.fc(self.tok_embeddings(x))
        return out, None, None


# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, seq_len, vocab_size):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __len__(self):
        return 198

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(idx)
        return ((torch.rand(self.seq_len + 1, generator=generator) * self.vocab_size).long(),)


# Unit test
class TestGradientAccumulation(unittest.TestCase):
    def test_accumulation(self):
        args = {
            "device": "cpu",
            "precision": "fp16",
            "accum_freq": 1,
            "seq_len": 9,
            "vocab_size": 10,
            "batch_size": 16,
            "log_logit_mean": False,
            "grad_clip_norm": 1.0,
            "skip_scheduler": True,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "wandb": False,
            "log_every_n_steps": 1,
            "target_mask_left": None,
            "target_mask_individual": None,
        }

        model1 = SimpleModel(vocab_size=args["vocab_size"])
        model2 = SimpleModel(vocab_size=args["vocab_size"])
        model2.load_state_dict(model1.state_dict())

        # Check if the weights are similar
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(
                torch.allclose(p1, p2, atol=1e-7),
                "Weights differ between accumulation modes.",
            )

        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.001)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001)

        loss_fn = torch.nn.CrossEntropyLoss()
        dataset = DummyDataset(seq_len=args["seq_len"], vocab_size=args["vocab_size"])
        dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)
        dataloader.num_batches = len(dataloader)
        dataloader.num_samples = len(dataloader) * args["batch_size"]
        # Train model1 without accumulation
        args["accum_freq"] = 2
        scaler = None  # GradScaler()
        data = Namespace(dataloader=dataloader, set_epoch=lambda x: None)

        train_one_epoch(
            model=model1,
            data={"train": data},
            loss=loss_fn,
            step=0,
            epoch=0,
            optimizer=optimizer1,
            scaler=scaler,
            scheduler=None,
            total_steps=-1,
            args=Namespace(**args),
        )
        # Train model2 with accumulation
        args["accum_freq"] = 1
        train_one_epoch(
            model=model2,
            data={"train": data},
            loss=loss_fn,
            step=0,
            epoch=0,
            optimizer=optimizer2,
            scaler=scaler,
            scheduler=None,
            total_steps=-1,
            args=Namespace(**args),
        )
        # Check if the weights are similar
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(
                torch.allclose(p1, p2, atol=1e-7),
                "Weights differ between accumulation modes.",
            )


if __name__ == "__main__":
    unittest.main()
