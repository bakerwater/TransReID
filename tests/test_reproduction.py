import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from config import cfg as default_cfg
from utils.metrics import R1_mAP_eval, eval_func


class DatasetTests(unittest.TestCase):
    def test_local_configs_merge_with_yacs(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ('market1501', 'msmt17'):
            config = default_cfg.clone()
            config.merge_from_file(str(root / f'configs/local/{dataset}_transreid_stride.yml'))
            self.assertEqual(config.MODEL.DEVICE_ID, '0')
            self.assertEqual(config.DATASETS.NAMES, dataset)
            self.assertEqual(config.SOLVER.MAX_EPOCHS, 120)
            self.assertEqual(config.SOLVER.IMS_PER_BATCH, 64)

    def test_converted_msmt_preserves_evaluation_ids_and_camera_15(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'MSMT17/images'
            for split in ('bounding_box_train', 'query', 'bounding_box_test'):
                (base / split).mkdir(parents=True)
            for name in ('0007_c1_0000.jpg', '0042_c15_0001.jpg'):
                (base / 'bounding_box_train' / name).touch()
            (base / 'query/0042_c1_0000.jpg').touch()
            (base / 'bounding_box_test/0042_c15_0001.jpg').touch()
            dataset = MSMT17(root=directory, verbose=False)
            self.assertEqual([r[1] for r in dataset.train], [0, 1])
            self.assertEqual(dataset.train[1][2], 14)
            self.assertEqual(dataset.query[0][1], 42)
            self.assertEqual(dataset.gallery[0][1], 42)
            self.assertEqual(dataset.num_train_pids, 2)

    def test_original_msmt_list_layout_still_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'MSMT17'
            (base / 'train').mkdir(parents=True)
            (base / 'test').mkdir()
            (base / 'list_train.txt').write_text('0000/0000_0000_01_0000.jpg 0\n')
            (base / 'list_val.txt').write_text('0000/0000_0001_15_0000.jpg 0\n')
            (base / 'list_query.txt').write_text('0000/0000_0000_01_0000.jpg 0\n')
            (base / 'list_gallery.txt').write_text('0000/0000_0001_15_0000.jpg 0\n')
            dataset = MSMT17(root=directory, verbose=False)
            self.assertEqual(len(dataset.train), 2)
            self.assertEqual(dataset.gallery[0][2], 14)

    def test_nested_market_ignores_junk_gallery_images(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'market1501/images'
            for split in ('bounding_box_train', 'query', 'bounding_box_test'):
                (base / split).mkdir(parents=True)
                (base / split / '0042_c1s1_000001_00.jpg').touch()
            (base / 'bounding_box_test/-1_c2s1_000002_00.jpg').touch()
            dataset = Market1501(root=directory, verbose=False)
            self.assertEqual(dataset.train[0][1], 0)
            self.assertEqual(dataset.query[0][1], 42)
            self.assertEqual(len(dataset.gallery), 1)


class MetricsTests(unittest.TestCase):
    def test_streaming_matches_full_matrix_with_invalid_queries(self):
        generator = torch.Generator().manual_seed(42)
        features = torch.randn(29, 12, generator=generator)
        query_ids = [999, 1, 1, 2, 3]
        gallery_ids = list(range(1, 7)) * 4
        query_cams = [0, 0, 1, 0, 1]
        gallery_cams = [0] * 12 + [1] * 12
        evaluator = R1_mAP_eval(5, max_rank=10)
        evaluator.reset()
        evaluator.update((features, query_ids + gallery_ids, query_cams + gallery_cams))
        full = evaluator.compute()
        for size in (1, 2, 3, 128):
            streamed = evaluator.compute(return_distmat=False, query_chunk_size=size)
            np.testing.assert_allclose(full[0], streamed[0], atol=1e-7)
            self.assertAlmostEqual(full[1], streamed[1], places=12)
            self.assertIsNone(streamed[2])

    def test_same_person_same_camera_is_excluded(self):
        cmc, mean_ap = eval_func(np.array([[0., .1, .2]]), np.array([1]),
                                np.array([1, 2, 1]), np.array([0]),
                                np.array([0, 1, 1]), max_rank=2)
        np.testing.assert_array_equal(cmc, [0., 1.])
        self.assertEqual(mean_ap, .5)


if __name__ == '__main__':
    unittest.main()
