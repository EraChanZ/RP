import pytest
import torch

# Import your config if used:
from config import cfg

# Import CoralTrainer (and parse_args if needed) from your code
from InterWild.main.train_coral import CoralTrainer

@pytest.fixture
def mock_args():
    """
    Create a mock object for the arguments normally parsed
    by parse_args(), with minimal VRAM usage in mind.
    """
    class MockArgs:
        gpu_ids = '0'
        continue_train = False
        source_ckpt = ''
        batch_size = 2
        amp = False
        grad_accumulation_steps = 1
        # Optionally, limit number of total pairs/batches
        max_batches = 2

    return MockArgs()

def test_coral_trainer_setup_no_forward(mock_args):
    """
    Test CoralTrainer's setup and DataLoader creation without
    performing any forward or backward pass to avoid high VRAM usage.
    """
    # We can safely run on CPU or a single GPU with minimal batch size:
    cfg.num_gpus = 1
    cfg.train_batch_size = mock_args.batch_size
    cfg.end_epoch = 1  # Just 1 epoch for quick test
    cfg.num_thread = 0 # Avoid multi-process overhead

    trainer = CoralTrainer(mock_args)
    trainer._make_batch_generator()
    trainer._make_model()

    # The trainer should have a batch generator and model
    assert trainer.batch_generator is not None, "batch_generator must be created."
    assert trainer.itr_per_epoch > 0, \
        "itr_per_epoch must be > 0; indicates there's data to iterate over."
    assert trainer.model is not None, "model must be built."

    # Basic param check
    param_count = sum(p.numel() for p in trainer.model.parameters())
    print(f"[DEBUG] Model param count: {param_count}")

    print("[VERBOSE TEST PASSED] CoralTrainer setup is correct without running forward/backward.")