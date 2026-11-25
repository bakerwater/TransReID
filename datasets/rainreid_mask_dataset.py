import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .bases import read_image


def load_reliability_annotations(dataset, annotation_csv, mask_root):
    """Read mask supervision splits from a manifest tied to the ReID train set."""
    if not annotation_csv or not mask_root:
        raise ValueError('RainReID reliability requires both the annotation CSV and mask root')
    mask_root = Path(mask_root).expanduser()
    annotation_csv = Path(annotation_csv).expanduser()
    train_images = {Path(path).resolve(): (path, pid) for path, pid, _, _ in dataset.train}
    annotations = {'train': {}, 'test': {}}
    split_pids = {'train': set(), 'test': set()}
    seen = set()

    with annotation_csv.open(encoding='utf-8-sig', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        columns = {'split', 'image_path', 'body_mask', 'umbrella_mask'}
        if not columns.issubset(reader.fieldnames or ()):
            raise ValueError(f'{annotation_csv} requires columns {", ".join(sorted(columns))}')
        for row in reader:
            split = (row['split'] or '').strip()
            image_path = (dataset.dataset_dir / (row['image_path'] or '').strip()).resolve()
            if split not in annotations:
                raise ValueError(f'Unknown reliability split {split!r} on line {reader.line_num} of {annotation_csv}')
            if image_path not in train_images:
                raise ValueError(f'Reliability annotation image is outside the RainReID training split: {image_path}')
            if image_path in seen:
                raise ValueError(f'Duplicate reliability annotation for {image_path}')

            body = (row['body_mask'] or '').strip()
            umbrella = (row['umbrella_mask'] or '').strip()
            if not body:
                raise ValueError(f'Missing body mask for {image_path}')
            body_path = mask_root / body
            umbrella_path = mask_root / umbrella if umbrella else None
            with Image.open(image_path) as image:
                _read_binary_mask(body_path, image.size)
                if umbrella_path is not None:
                    _read_binary_mask(umbrella_path, image.size)

            sample_path, pid = train_images[image_path]
            annotations[split][sample_path] = (body_path, umbrella_path)
            split_pids[split].add(pid)
            seen.add(image_path)

    if not all(annotations.values()):
        raise ValueError(f'{annotation_csv} must contain train and held-out test mask annotations')
    if split_pids['train'] & split_pids['test']:
        raise ValueError(f'Train and held-out mask annotations share identities in {annotation_csv}')
    return annotations


def _read_binary_mask(path, image_size):
    with Image.open(path) as mask:
        if mask.format != 'PNG' or mask.mode not in ('1', 'L'):
            raise ValueError(f'Expected a binary grayscale PNG mask: {path}')
        if mask.size != image_size:
            raise ValueError(f'Mask size {mask.size} does not match image size {image_size}: {path}')
        mask = mask.convert('L')
        colors = mask.getcolors(maxcolors=2)
        if colors is None or any(value not in (0, 255) for _, value in colors):
            raise ValueError(f'Mask must contain only black and white pixels: {path}')
        return mask


class RainReIDMaskDataset(Dataset):
    """Keep the full ReID train set and expose mask targets only for its selected partition."""

    def __init__(self, samples, transform, annotations, split):
        self.annotations = annotations[split]
        self.annotation_counts = {key: len(items) for key, items in annotations.items()}
        self.samples = samples if split == 'train' else [sample for sample in samples if sample[0] in self.annotations]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, pid, camera, view = self.samples[index]
        image = read_image(path)
        annotation = self.annotations.get(path)
        if annotation is not None:
            body_path, umbrella_path = annotation
            body = _read_binary_mask(body_path, image.size)
            umbrella = (_read_binary_mask(umbrella_path, image.size) if umbrella_path is not None
                        else Image.new('L', image.size))
            image, (body, umbrella) = self.transform(image, (body, umbrella))
        else:
            image, _ = self.transform(image)
            body = umbrella = image.new_zeros((1, *image.shape[-2:]))
        return image, pid, camera, view, body, umbrella, annotation is not None
