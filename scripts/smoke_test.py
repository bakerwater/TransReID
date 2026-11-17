"""Exercise real 64-image AMP training batches and FP32 inference on each dataset."""
import gc
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import torch
from config import cfg as default_cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from solver import make_optimizer
from train import set_seed


def main():
    os.chdir(root)
    torch.set_num_threads(8)
    report = {}
    for dataset in ('market1501', 'msmt17'):
        config = default_cfg.clone()
        config.merge_from_file(str(root / f'configs/local/{dataset}_transreid_stride.yml'))
        config.freeze()
        set_seed(config.SOLVER.SEED)
        train, _, validation, _, classes, cameras, views = make_dataloader(config)
        model = make_model(config, classes, cameras, views).cuda()
        loss_fn, center = make_loss(config, classes)
        optimizer, _ = make_optimizer(config, model, center)
        scaler = torch.amp.GradScaler('cuda')
        torch.cuda.reset_peak_memory_stats()
        model.train()
        losses = []
        iterator = iter(train)
        start = time.monotonic()
        for _ in range(3):
            images, labels, camera, view = [value.cuda() for value in next(iterator)]
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                score, feature = model(images, labels, cam_label=camera, view_label=view)
                loss = loss_fn(score, feature, labels, camera)
            assert torch.isfinite(loss).item(), f'{dataset}: nonfinite loss'
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        torch.cuda.synchronize()
        training_seconds = time.monotonic() - start
        model.eval()
        val_iterator = iter(validation)
        images, _, _, camera, view, _ = next(val_iterator)
        with torch.no_grad():
            feature = model(images.cuda(), cam_label=camera.cuda(), view_label=view.cuda())
        assert feature.shape == (config.TEST.IMS_PER_BATCH, 3840), feature.shape
        assert torch.isfinite(feature).all().item()
        report[dataset] = {'batch_size': config.SOLVER.IMS_PER_BATCH,
                           'losses': losses, 'three_batches_seconds': training_seconds,
                           'inference_shape': list(feature.shape),
                           'peak_gpu_memory_gib': torch.cuda.max_memory_allocated() / 2 ** 30}
        print(json.dumps({dataset: report[dataset]}, indent=2), flush=True)
        del iterator, val_iterator, train, validation, model, optimizer, center
        del images, labels, camera, view, score, feature, loss, loss_fn, scaler
        gc.collect()
        torch.cuda.empty_cache()
    (root / 'logs/smoke_report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
