import re
from pathlib import Path

from .bases import BaseImageDataset


class RainReID(BaseImageDataset):
    """Load the released RainReID full-search train, query, and gallery splits."""

    dataset_dir = 'RainReID'
    filename_pattern = re.compile(r'(\d+)_c(\d+)_\d+\.(?:jpg|jpeg)', re.IGNORECASE)

    def __init__(self, root='', verbose=True):
        root = Path(root).expanduser()
        self.dataset_dir = root if root.name == self.dataset_dir else root / self.dataset_dir
        self.train_dir = self.dataset_dir / 'bounding_box_train'
        self.query_dir = self.dataset_dir / 'query'
        self.gallery_dir = self.dataset_dir / 'bounding_box_test'

        train = self._read_split(self.train_dir)
        query = self._read_split(self.query_dir)
        gallery = self._read_split(self.gallery_dir)

        train_pids = sorted({pid for _, pid, _ in train})
        pid_to_label = {pid: label for label, pid in enumerate(train_pids)}
        # Camera 4 is absent from the released training split. Share one camera
        # mapping across splits so the indices agree during cross-camera retrieval.
        cameras = sorted({cam for split in (train, query, gallery) for _, _, cam in split})
        cam_to_label = {cam: label for label, cam in enumerate(cameras)}
        self.num_cams = len(cameras)

        # The filename suffix identifies the crop. The dataset has no view labels.
        self.train = [(path, pid_to_label[pid], cam_to_label[cam], 0) for path, pid, cam in train]
        self.query = [(path, pid, cam_to_label[cam], 0) for path, pid, cam in query]
        self.gallery = [(path, pid, cam_to_label[cam], 0) for path, pid, cam in gallery]

        (self.num_train_pids, self.num_train_imgs, self.num_train_cams,
         self.num_train_vids) = self.get_imagedata_info(self.train)
        (self.num_query_pids, self.num_query_imgs, self.num_query_cams,
         self.num_query_vids) = self.get_imagedata_info(self.query)
        (self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams,
         self.num_gallery_vids) = self.get_imagedata_info(self.gallery)

        if verbose:
            print('=> RainReID loaded')
            self.print_dataset_statistics(self.train, self.query, self.gallery)

    def _read_split(self, directory):
        if not directory.is_dir():
            raise FileNotFoundError(f'RainReID split directory does not exist: {directory}')

        samples = []
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg'}:
                continue
            match = self.filename_pattern.fullmatch(path.name)
            if match is None:
                raise ValueError(f'Invalid RainReID filename: {path}')
            pid, cam = map(int, match.groups())
            samples.append((str(path), pid, cam))

        if not samples:
            raise ValueError(f'RainReID split has no images: {directory}')
        return samples
