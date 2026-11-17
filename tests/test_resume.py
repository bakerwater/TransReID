"""Verify optimizer, scheduler, AMP and RNG restoration across an epoch boundary."""
import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import cfg as default_cfg
from processor import do_train
from solver.scheduler_factory import create_scheduler
from train import set_seed


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(.2))
        self.classifier = nn.Linear(8, 2)

    def forward(self, images, labels=None, cam_label=None, view_label=None):
        feature = self.features(images)
        return (self.classifier(feature), feature) if self.training else feature


@unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
class ResumeTests(unittest.TestCase):
    def run_training(self, output, epochs, seed=42, resume=''):
        set_seed(seed)
        config = default_cfg.clone()
        config.MODEL.NAME = 'small'
        config.MODEL.METRIC_LOSS_TYPE = 'triplet'
        config.SOLVER.MAX_EPOCHS = 2
        config.SOLVER.CHECKPOINT_PERIOD = 1
        config.SOLVER.EVAL_PERIOD = 1
        config.SOLVER.RESUME = resume
        config.OUTPUT_DIR = str(output)
        generator = torch.Generator().manual_seed(123)
        images = torch.randn(16, 4, generator=generator)
        ids = torch.arange(16) % 2
        cameras = torch.ones(16, dtype=torch.int64)
        cameras[:2] = 0
        views = torch.ones(16, dtype=torch.int64)
        train = DataLoader(TensorDataset(images, ids, cameras, views), batch_size=8, shuffle=True)
        validation = DataLoader(TensorDataset(images, ids, cameras, cameras, views,
                                            torch.arange(16)), batch_size=8)
        model = SmallModel().cuda()
        center = nn.Linear(4, 2).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        center_optimizer = torch.optim.SGD(center.parameters(), lr=.01)
        scheduler = create_scheduler(config, optimizer)
        # Keep the same two-epoch schedule while simulating an interruption after epoch one.
        config.SOLVER.MAX_EPOCHS = epochs
        loss_fn = lambda score, feature, target, camera: nn.functional.cross_entropy(score, target)
        do_train(config, model, center, train, validation, optimizer, center_optimizer,
                 scheduler, loss_fn, 2, 0)
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def test_resumed_training_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            uninterrupted = Path(directory) / 'uninterrupted'
            restarted = Path(directory) / 'restarted'
            uninterrupted.mkdir()
            restarted.mkdir()
            expected = self.run_training(uninterrupted, 2)
            self.run_training(restarted, 1)
            actual = self.run_training(restarted, 2, seed=999,
                                       resume=str(restarted / 'latest_training.pth'))
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
