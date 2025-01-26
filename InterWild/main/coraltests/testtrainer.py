import unittest
import torch
import os.path as osp
import sys

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))

from train_coral import CoralTrainer, MSCOCODataset, InfraredHandDataset
from config import cfg  # Assuming this is accessible
import argparse

class TestCoralTrainer(unittest.TestCase):

    def setUp(self):
        """Setup a minimal CoralTrainer with mock data."""
        # Mock arguments
        self.args = argparse.Namespace(gpu_ids='0', continue_train=False, source_ckpt='')
        cfg.set_args(self.args.gpu_ids, self.args.continue_train)
        cfg.train_batch_size = 2  # Small batch size for testing
        cfg.num_thread = 1

        self.trainer = CoralTrainer(self.args)

        # Create mock datasets
        self.source_dataset = MSCOCODataset(transform=None)
        self.target_dataset = InfraredHandDataset(
            root_path=r'C:\Users\vladi\RP\Research\IR_videos\every60thframe',
            transform=None
        )

    def test_batch_generator_shapes(self):
        """Test if batch generator produces correct shapes."""
        self.trainer._make_batch_generator()  # Create the batch generator

        for batch_idx, batch_data in enumerate(self.trainer.batch_generator):
            (source_inputs, source_targets, source_meta_info,
             target_inputs, target_meta_info) = batch_data

            # Check batch size
            self.assertEqual(source_inputs['img'].shape[0], cfg.num_gpus * cfg.train_batch_size)

            # Check data types and shapes (add more assertions based on your data)
            self.assertTrue(isinstance(source_inputs['img'], torch.Tensor))
            self.assertTrue(isinstance(target_inputs['img'], torch.Tensor))
            # Example: check if 'joint_img' exists in source_targets and is a Tensor
            if 'joint_img' in source_targets:
                self.assertTrue(isinstance(source_targets['joint_img'], torch.Tensor))

            # Check for meta info (if applicable)
            if source_meta_info:
                self.assertTrue(isinstance(source_meta_info, dict))
            if target_meta_info:
                self.assertTrue(isinstance(target_meta_info, dict))

            break  # Only check the first batch for brevity

    def test_batch_generator_content(self):
        """Test if batch generator produces expected content."""
        self.trainer._make_batch_generator()

        for batch_idx, batch_data in enumerate(self.trainer.batch_generator):
            (source_inputs, source_targets, source_meta_info,
             target_inputs, target_meta_info) = batch_data

            # Check if source data comes from MSCOCODataset
            # This is a bit tricky without more specific identifiers in your data.
            # You might need to add some unique markers to your datasets to distinguish them.
            # For now, we can check if the keys match what's expected from MSCOCODataset.
            expected_source_keys = {'img'}  # Add more keys if your MSCOCODataset returns more
            self.assertTrue(set(source_inputs.keys()).issuperset(expected_source_keys))

            # Check if target data comes from InfraredHandDataset
            expected_target_keys = {'img'}
            self.assertTrue(set(target_inputs.keys()).issuperset(expected_target_keys))

            break

    def test_batch_generator_length(self):
        """Test if the batch generator has the correct length."""
        self.trainer._make_batch_generator()

        # Calculate expected number of batches
        expected_num_batches = min(len(self.source_dataset), len(self.target_dataset)) // (cfg.num_gpus * cfg.train_batch_size)

        actual_num_batches = 0
        for _ in self.trainer.batch_generator:
            actual_num_batches += 1

        self.assertEqual(actual_num_batches, expected_num_batches)

if __name__ == '__main__':
    unittest.main()