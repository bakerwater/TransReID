#!/usr/bin/env python3
"""
Extract train and query+gallery features from the official TransReID codebase.

Run from the TransReID repository root:
python tools/extract_camera_features.py \
  --config_file configs/Market/vit_transreid_stride.yml \
  --output_dir outputs/stage1/market \
  --mode both \
  TEST.WEIGHT /absolute/path/to/transformer_120.pth

Modes:
  native  : normal TransReID inference, preserving the configured SIE camera embedding.
  neutral : set model.base.sie_xishu=0 during inference, disabling the explicit SIE addition.
  both    : extract both modes.

The "neutral" pass is a diagnostic for estimating appearance relationships without
injecting the camera-specific SIE embedding. It is NOT the official baseline score.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from datasets import make_dataloader
from model import make_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--mode", choices=("native", "neutral", "both"), default="both"
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def extract_loader(model, loader, device):
    all_features = []
    all_pids = []
    all_camids = []
    all_viewids = []
    all_names = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Official TransReID val_collate_fn:
            # imgs, pids, camids, camids_batch, viewids, img_paths
            imgs, pids, camids, camids_batch, viewids, img_paths = batch

            imgs = imgs.to(device, non_blocking=True)
            camids_batch = torch.as_tensor(camids_batch, dtype=torch.long, device=device)
            viewids = torch.as_tensor(viewids, dtype=torch.long, device=device)

            features = model(imgs, cam_label=camids_batch, view_label=viewids)
            if isinstance(features, (tuple, list)):
                raise RuntimeError(
                    "Expected a single eval feature tensor, but model returned "
                    f"{type(features)}. Check MODEL.JPM / model.eval() behavior."
                )

            all_features.append(features.detach().float().cpu())
            all_pids.extend(np.asarray(pids, dtype=np.int64).tolist())
            all_camids.extend(np.asarray(camids, dtype=np.int64).tolist())
            all_viewids.extend(torch.as_tensor(viewids).cpu().numpy().astype(np.int64).tolist())
            all_names.extend([str(x) for x in img_paths])

            if (batch_idx + 1) % 50 == 0:
                print(f"  extracted {batch_idx + 1}/{len(loader)} batches")

    return {
        # Store half precision to reduce disk use; analysis script converts to float32.
        "features": torch.cat(all_features, dim=0).half(),
        "pids": torch.tensor(all_pids, dtype=torch.long),
        "camids": torch.tensor(all_camids, dtype=torch.long),
        "viewids": torch.tensor(all_viewids, dtype=torch.long),
        "img_names": all_names,
    }


def main():
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    if not cfg.TEST.WEIGHT:
        raise ValueError(
            "TEST.WEIGHT is empty. Pass the trained checkpoint after the script options."
        )
    weight_path = Path(cfg.TEST.WEIGHT).expanduser()
    if not weight_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    # Follow the official test.py device selection behavior.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA is unavailable; extraction may be very slow.")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        _train_loader,
        train_loader_normal,
        val_loader,
        num_query,
        num_classes,
        camera_num,
        view_num,
    ) = make_dataloader(cfg)

    print("Dataset:", cfg.DATASETS.NAMES)
    print("Train images for feature extraction:", len(train_loader_normal.dataset))
    print("Query images:", num_query)
    print("Query + gallery images:", len(val_loader.dataset))
    print("Train identities:", num_classes)
    print("Camera count:", camera_num)
    print("View count:", view_num)
    print("Checkpoint:", weight_path)

    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num
    )
    model.load_param(str(weight_path))
    model.to(device)
    model.eval()

    if not hasattr(model, "base") or not hasattr(model.base, "sie_xishu"):
        raise RuntimeError(
            "Could not find model.base.sie_xishu. This script expects the official "
            "TransReID ViT implementation. Inspect your local model before proceeding."
        )

    original_sie_xishu = float(model.base.sie_xishu)
    if args.mode == "both":
        modes = ("native", "neutral")
    else:
        modes = (args.mode,)

    for mode in modes:
        if mode == "native":
            model.base.sie_xishu = original_sie_xishu
        else:
            # Disable the explicit camera/view SIE additive term for a diagnostic pass.
            # Weights are not changed; only the SIE multiplier is set to zero.
            model.base.sie_xishu = 0.0

        print(f"\n=== Extracting mode: {mode}; SIE_COE={model.base.sie_xishu} ===")
        train_pack = extract_loader(model, train_loader_normal, device)
        test_pack = extract_loader(model, val_loader, device)
        test_pack["num_query"] = int(num_query)

        metadata = {
            "dataset": str(cfg.DATASETS.NAMES),
            "config_file": str(args.config_file),
            "checkpoint": str(weight_path),
            "mode": mode,
            "sie_xishu": float(model.base.sie_xishu),
            "neck_feat": str(cfg.TEST.NECK_FEAT),
            "feat_norm_in_evaluator": str(cfg.TEST.FEAT_NORM),
            "num_query": int(num_query),
            "camera_num": int(camera_num),
            "num_train_ids": int(num_classes),
        }
        train_pack["metadata"] = metadata
        test_pack["metadata"] = metadata

        train_path = output_dir / f"train_features_{mode}.pt"
        test_path = output_dir / f"test_features_{mode}.pt"
        torch.save(train_pack, train_path)
        torch.save(test_pack, test_path)

        print("Saved:", train_path, train_pack["features"].shape)
        print("Saved:", test_path, test_pack["features"].shape)

    model.base.sie_xishu = original_sie_xishu
    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
