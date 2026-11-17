"""Check CUDA, the pretrained checksum, and both complete dataset splits."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import numpy as np
from PIL import Image
import timm
import torch
import torchvision
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17


def verify_image(path):
    with Image.open(path) as image:
        image.verify()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-images', action='store_true')
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA is unavailable'
    assert torch.version.cuda, 'A CPU-only PyTorch build was installed'
    value = torch.randn(512, 512, device='cuda')
    assert torch.isfinite(value @ value.T).all().item()
    report = {'python': sys.version, 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
              'numpy': np.__version__, 'timm': timm.__version__,
              'gpu': torch.cuda.get_device_name(0), 'datasets': {}}
    checkpoint = root / 'pretrained/jx_vit_base_p16_224-80ecf9dd.pth'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == '80ecf9dd5e3a58895e959af554c5666c4e7b4da4410de4f1f2b0025e93435d8c'
    report['pretrained_sha256'] = digest
    for cls, name, expected in [(Market1501, 'market1501', (12936, 3368, 15913)),
                                (MSMT17, 'msmt17', (32621, 11659, 82161))]:
        dataset = cls(root=str(root / 'data'))
        splits = [dataset.train, dataset.query, dataset.gallery]
        counts = tuple(map(len, splits))
        assert counts == expected, f'{name}: expected {expected}, found {counts}'
        assert {x[1] for x in dataset.train} == set(range(dataset.num_train_pids))
        gallery_cameras = defaultdict(set)
        for _, pid, camera, _ in dataset.gallery:
            gallery_cameras[pid].add(camera)
        valid_queries = sum(bool(gallery_cameras[pid] - {camera})
                            for _, pid, camera, _ in dataset.query)
        assert valid_queries > 0, f'{name}: no valid cross-camera matches'
        assert dataset.num_train_pids == (751 if name == 'market1501' else 1041)
        assert dataset.num_train_cams == (6 if name == 'market1501' else 15)
        paths = [record[0] for split in splits for record in split]
        assert all(Path(path).is_file() for path in paths)
        if args.verify_images:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for _ in pool.map(verify_image, paths):
                    pass
        report['datasets'][name] = dict(zip(('train', 'query', 'gallery'), counts))
        report['datasets'][name]['train_ids'] = dataset.num_train_pids
        report['datasets'][name]['cameras'] = dataset.num_train_cams
        report['datasets'][name]['valid_queries'] = valid_queries
        report['datasets'][name]['images_verified'] = args.verify_images
    (root / 'logs').mkdir(exist_ok=True)
    (root / 'logs/setup_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
