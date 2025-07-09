import datetime
import glob
import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_model

# Set environment variables
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "1800"


@dataclass
class Hyperparameters(Coqpit):
    """Training hyperparameters configuration."""

    run_name: str = field(
        default="nano_gpt+rms_norm+geglu+gqa+softcap",
        metadata={"help": "Name for this training run"}
    )
    compile_model: bool = field(
        default=False,
        metadata={"help": "Compile model with torch.compile for better performance"}
    )
    input_bin: str = field(
        default="../data/fineweb10B/fineweb_train_*.bin",
        metadata={"help": "Pattern for training data files (BPE tokenized)"}
    )
    input_val_bin: str = field(
        default="../data/fineweb10B/fineweb_val_*.bin",
        metadata={"help": "Pattern for validation data files (BPE tokenized)"}
    )
    byte_level_training: bool = field(
        default=False,
        metadata={"help": "Enable byte-level training mode"}
    )
    text_data_dir: str = field(
        default="../data/fineweb10B_text",
        metadata={"help": "Directory containing text data (.jsonl files)"}
    )
    byte_data_dir: str = field(
        default="../data/fineweb10B_bytes",
        metadata={"help": "Directory containing byte data (.bin files)"}
    )
    batch_size: int = field(
        default=8 * 64,
        metadata={"help": "Total batch size across all devices"}
    )
    device_batch_size: int = field(
        default=32,
        metadata={"help": "Batch size per device"}
    )
    sequence_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length for training"}
    )
    num_iterations: int = field(
        default=5100,
        metadata={"help": "Number of training iterations"}
    )
    learning_rate: float = field(
        default=0.001,
        metadata={"help": "Learning rate for optimizer"}
    )
    warmup_iters: int = field(
        default=250,
        metadata={"help": "Number of warmup iterations"}
    )
    warmdown_iters: int = field(
        default=2000,
        metadata={"help": "Number of warmdown iterations"}
    )
    weight_decay: float = field(
        default=0,
        metadata={"help": "Weight decay coefficient"}
    )
    val_loss_every: int = field(
        default=125,
        metadata={"help": "Validate every N iterations"}
    )
    val_tokens: int = field(
        default=10485760,
        metadata={"help": "Number of tokens for validation"}
    )
    save_every: int = field(
        default=5000,
        metadata={"help": "Save checkpoint every N iterations"}
    )
    keep_last_n_checkpoints: int = field(
        default=1,
        metadata={"help": "Number of recent checkpoints to keep"}
    )
    save_best_model: bool = field(
        default=True,
        metadata={"help": "Save best model based on validation loss"}
    )
    precision: str = field(
        default="bfloat16",
        metadata={"help": "Training precision (float32 or bfloat16)"}
    )


class TeeLogger:
    """Logger that writes to both terminal and file."""

    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        return hasattr(self.terminal, "isatty") and self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


class DataShard:
    """Handles loading and validation of data shards."""

    MAGIC_NUMBER = 20240520
    HEADER_SIZE = 256 * 4
    VERSION = 1

    @staticmethod
    def peek_data_shard(filename: str) -> int:
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(DataShard.HEADER_SIZE), dtype=np.int32)

        if header[0] != DataShard.MAGIC_NUMBER:
            print("ERROR: magic number mismatch in the data .bin file!")
            print("---> HINT: Are you passing in a correct file with --input_bin?")
            print(
                "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
            )
            print(
                "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
            )
            exit(1)
        assert header[1] == DataShard.VERSION, "Unsupported version"
        return header[2]

    @staticmethod
    def load_data_shard(filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(DataShard.HEADER_SIZE), dtype=np.int32)
            assert header[0] == DataShard.MAGIC_NUMBER, "Magic number mismatch"
            assert header[1] == DataShard.VERSION, "Unsupported version"
            ntok = header[2]
            tokens = np.frombuffer(f.read(), dtype=np.uint16)

        assert len(tokens) == ntok, "Token count mismatch"
        return tokens


class ByteDataShard:
    """Handles loading and validation of byte-level data shards."""

    MAGIC_NUMBER = 20240520
    HEADER_SIZE = 256 * 4
    VERSION = 1

    @staticmethod
    def peek_byte_data_shard(filename: str) -> int:
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(ByteDataShard.HEADER_SIZE), dtype=np.int32)

        if header[0] != ByteDataShard.MAGIC_NUMBER:
            print("ERROR: magic number mismatch in the byte data .bin file!")
            print("---> HINT: Are you passing in a correct byte-level data file?")
            print("---> HINT: For byte-level training, use fineweb_bytes.py to generate data")
            exit(1)
        assert header[1] == ByteDataShard.VERSION, "Unsupported version"
        return header[2]

    @staticmethod
    def load_byte_data_shard(filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(ByteDataShard.HEADER_SIZE), dtype=np.int32)
            assert header[0] == ByteDataShard.MAGIC_NUMBER, "Magic number mismatch"
            assert header[1] == ByteDataShard.VERSION, "Unsupported version"
            nbytes = header[2]
            bytes_data = np.frombuffer(f.read(), dtype=np.uint8)

        assert len(bytes_data) == nbytes, "Byte count mismatch"
        return bytes_data




class DistributedDataLoader:
    """Distributed data loader for handling multiple data shards."""

    def __init__(
        self,
        filename_pattern: str,
        batch_size: int,
        seq_length: int,
        process_rank: int,
        num_processes: int,
        byte_level_training: bool = False,
    ):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.byte_level_training = byte_level_training

        self.files = sorted(glob.glob(filename_pattern))
        if not self.files:
            raise ValueError(f"No files found matching pattern: {filename_pattern}")

        self.ntok_total = self._validate_shards()
        self.reset()

    def _load_shard_data(self, filename: str) -> np.ndarray:
        """Load data from a shard file based on training mode and file type."""
        if self.byte_level_training:
            if filename.endswith('.bin'):
                return ByteDataShard.load_byte_data_shard(filename)
            else:
                raise ValueError(f"Byte-level training only supports .bin files, got: {filename}")
        else:
            return DataShard.load_data_shard(filename)

    def _get_shard_size(self, filename: str) -> int:
        """Get the size of a shard file based on training mode and file type."""
        if self.byte_level_training:
            if filename.endswith('.bin'):
                return ByteDataShard.peek_byte_data_shard(filename)
            else:
                raise ValueError(f"Byte-level training only supports .bin files, got: {filename}")
        else:
            return DataShard.peek_data_shard(filename)

    def _validate_shards(self) -> int:
        ntok_total = 0
        min_required = self.num_processes * self.batch_size * self.seq_length + 1

        for fname in self.files:
            shard_ntok = self._get_shard_size(fname)
            assert shard_ntok >= min_required, f"Shard {fname} too small"
            ntok_total += int(shard_ntok)
        return ntok_total

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.batch_size * self.seq_length
        self.tokens = self._load_shard_data(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.batch_size * self.seq_length
        self.tokens = self._load_shard_data(self.files[self.current_shard])

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.batch_size, self.seq_length
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        # Convert to tensor - both bytes and tokens need to be long for model input
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)

        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        # For debugging: print max value to check data range
        if self.byte_level_training:
            assert x.max().item() < 256, f"Byte value out of range: {x.max().item()}"
        # print(x.max().item())

        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()

        return x.cuda(), y.cuda()


class Trainer:
    """Main trainer class handling the training loop and validation."""

    def __init__(
        self, args: Hyperparameters, model_name: str, config_path: Optional[str] = None
    ):
        self.args = args

        self.setup_distributed()
        self.setup_model(model_name, config_path)
        self.setup_optimization()
        self.setup_data_loaders()
        if self.is_master:
            self.setup_logging()

    def setup_distributed(self):
        """Initialize distributed training setup."""
        assert torch.cuda.is_available()
        dist.init_process_group(
            backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=30)
        )
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = f"cuda:{self.local_rank}"
        self.is_master = self.rank == 0
        torch.cuda.set_device(self.device)

    def setup_model(self, model_name: str, config_path: Optional[str]):
        """Initialize model and move to GPU."""
        model_config, model_cls = get_model(model_name)
        self.model_config = model_config()

        if config_path:
            self.model_config.load_json(config_path)

        # Override Hyperparameters with matching model_config attributes
        overridden_params = []
        for key, value in self.model_config.to_dict().items():
            if hasattr(self.args, key):
                old_value = getattr(self.args, key)
                if old_value != value:
                    overridden_params.append((key, old_value, value))
                setattr(self.args, key, value)

        # Log overridden parameters (only on master process)
        if overridden_params and self.is_master:
            print("Hyperparameters overridden by model config:")
            for key, old_val, new_val in overridden_params:
                print(f"  {key}: {old_val} -> {new_val}")

        # Validate vocab size for byte-level training
        if self.args.byte_level_training:
            if hasattr(self.model_config, 'vocab_size') and self.model_config.vocab_size != 256:
                raise ValueError(
                    f"Byte-level training requires vocab_size=256, but model config has vocab_size={self.model_config.vocab_size}. "
                    f"Please update your model configuration for byte-level training."
                )
            elif hasattr(self.model_config, 'vocab_size'):
                print(f"✓ Vocab size validation passed: {self.model_config.vocab_size} (suitable for byte-level training)")

        self.model = model_cls(self.model_config).cuda()

        if self.args.compile_model:
            print("Compiling the model...")
            self.model = torch.compile(self.model)
            print("✓ Model compilation passed")

        self.model = DDP(
            self.model, device_ids=[self.local_rank], find_unused_parameters=True
        )
        self.raw_model = self.model.module
        self.ctx = autocast(
            device_type="cuda", dtype=torch.bfloat16 if self.args.precision == "bfloat16" else torch.float32
        )

    def setup_optimization(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.Adam(
            self.raw_model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=self.args.weight_decay,
            foreach=False,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self.get_lr_schedule()
        )

    def get_lr_schedule(self):
        """Create learning rate schedule function."""

        def schedule(it):
            if it < self.args.warmup_iters:
                return (it + 1) / self.args.warmup_iters
            elif it < self.args.num_iterations - self.args.warmdown_iters:
                return 1.0
            else:
                return (self.args.num_iterations - it) / self.args.warmdown_iters

        return schedule

    def setup_data_loaders(self):
        """Initialize training and validation data loaders."""
        if self.args.byte_level_training:
            if self.is_master:
                print("Setting up byte-level training data loaders...")

            # Try binary byte shards first, fallback to text shards
            train_pattern = os.path.join(self.args.byte_data_dir, "fineweb_train_*.bin")
            val_pattern = os.path.join(self.args.byte_data_dir, "fineweb_val_*.bin")

            # Check if binary files exist
            train_files = glob.glob(train_pattern)
            val_files = glob.glob(val_pattern)

            if not train_files:
                raise ValueError(f"No byte-level training files found at {train_pattern}. "
                               f"Please generate byte data using fineweb_bytes.py first.")

            # If validation files don't exist, use a subset of training files
            if not val_files:
                if self.is_master:
                    print(f"No validation files found at {val_pattern}")
                    print("Using training files for validation...")
                val_pattern = train_pattern

            if self.is_master:
                print(f"Using binary files for byte-level training")
                print(f"Training files: {len(glob.glob(train_pattern))}")
                print(f"Validation files: {len(glob.glob(val_pattern))}")
        else:
            train_pattern = self.args.input_bin
            val_pattern = self.args.input_val_bin

            if self.is_master:
                print("Setting up BPE token-level training data loaders...")
                print(f"Training pattern: {train_pattern}")
                print(f"Validation pattern: {val_pattern}")

        try:
            self.train_loader = DistributedDataLoader(
                train_pattern,
                self.args.device_batch_size,
                self.args.sequence_length,
                self.rank,
                self.world_size,
                self.args.byte_level_training,
            )

            self.val_loader = DistributedDataLoader(
                val_pattern,
                self.args.device_batch_size,
                self.args.sequence_length,
                self.rank,
                self.world_size,
                self.args.byte_level_training,
            )
        except Exception as e:
            if self.is_master:
                print(f"Error setting up data loaders: {e}")
                print("\nTroubleshooting:")
                if self.args.byte_level_training:
                    print("For byte-level training:")
                    print(f"- Ensure byte data directory exists: {self.args.byte_data_dir}")
                    print("- Run: cd ../data && python fineweb_bytes.py --version 10B")
                    print("- Check that .bin files are present in the byte_data_dir")
                else:
                    print("For BPE token training:")
                    print(f"- Check input_bin path: {self.args.input_bin}")
                    print(f"- Check input_val_bin path: {self.args.input_val_bin}")
                    print("- Run: cd ../data && python fineweb.py --version 10B")
            raise

        self.train_accumulation_steps = self.args.batch_size // (
            self.args.device_batch_size * self.world_size
        )

        self.val_steps = self.args.val_tokens // (
            self.args.device_batch_size * self.args.sequence_length * self.world_size
        )

        if self.is_master:
            print(f"Data loader setup complete:")
            print(f"  Total training tokens: {self.train_loader.ntok_total:,}")
            print(f"  Total validation tokens: {self.val_loader.ntok_total:,}")
            print(f"  Training accumulation steps: {self.train_accumulation_steps}")
            print(f"  Validation steps: {self.val_steps}")
            if self.args.byte_level_training:
                print(f"  Vocabulary size: 256 (byte-level)")
            else:
                print(f"  Vocabulary size: {getattr(self.model_config, 'vocab_size', 'Unknown')}")

    def setup_logging(self):
        """Initialize logging directory and files."""
        run_num = 0
        self.run_id = f"{self.args.run_name}_{run_num}"
        self.logdir = f"logs/{self.run_id}/"

        while os.path.exists(self.logdir):
            run_num += 1
            self.run_id = f"{self.args.run_name}_{run_num}"
            self.logdir = f"logs/{self.run_id}/"

        os.makedirs(self.logdir, exist_ok=True)
        self.logfile = f"logs/{self.run_id}.txt"

        sys.stdout = TeeLogger(self.logfile)
        self._log_initial_info()

    def _log_initial_info(self):
        """Log initial information about the training run."""

        print(
            f"Running pytorch {torch.__version__} compiled for CUDA {torch.version.cuda}"
        )

        print("=" * 100)
        print(f"Byte-level training: {'Enabled' if self.args.byte_level_training else 'Disabled'}")
        if self.args.byte_level_training:
            print(f"Text data directory: {self.args.text_data_dir}")
            print(f"Vocab size: 256 (bytes)")
        print("=" * 100)

    def validate(self) -> float:
        """Run validation loop and return validation loss."""
        self.model.eval()
        self.val_loader.reset()
        val_loss = 0.0

        with torch.no_grad():
            for _ in range(self.val_steps):
                x_val, y_val = self.val_loader.next_batch()
                with self.ctx:
                    _, loss = self.model(x_val, y_val)

                    if isinstance(loss, dict):
                        loss = loss["total"]
                    val_loss += loss.detach()

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        return val_loss / self.val_steps

    def save_checkpoint(self, step: int, val_loss: Optional[float] = None):
        """Save model checkpoint."""
        log = {
            "step": step,
            "model_config": self.model_config.to_dict(),
            "train_config": self.args.to_dict(),
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if val_loss is not None:
            log["val_loss"] = val_loss

        checkpoint_path = f"{self.logdir}/state_step{step:06d}.pt"
        torch.save(log, checkpoint_path)

        # Cleanup old checkpoints
        if self.args.keep_last_n_checkpoints > 0:
            checkpoints = sorted(glob.glob(f"{self.logdir}/state_step*.pt"))
            if len(checkpoints) > self.args.keep_last_n_checkpoints:
                for checkpoint in checkpoints[: -self.args.keep_last_n_checkpoints]:
                    os.remove(checkpoint)

    def save_best_model(self, step: int, val_loss: float):
        """Save model if it has the best validation loss so far."""
        log = {
            "step": step,
            "model_config": self.model_config.to_dict(),
            "train_config": self.args.to_dict(),
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }

        best_model_path = f"{self.logdir}/best_model_{step}.pt"
        torch.save(log, best_model_path)

        # Remove previous best model if exists
        for f in glob.glob(f"{self.logdir}/best_model_*.pt"):
            if f != best_model_path:
                os.remove(f)

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[Dict]]:
        """Execute a single training step."""
        self.model.train()
        metrics = None

        for i in range(1, self.train_accumulation_steps + 1):
            with self.ctx:
                _, loss = self.model(x, y)
                if isinstance(loss, dict):
                    metrics = {k: v for k, v in loss.items() if k != "total"}
                    loss = loss["total"]
                train_loss = loss.detach()

            x, y = self.train_loader.next_batch()

            if i < self.train_accumulation_steps:
                with self.model.no_sync():
                    loss.backward()
            else:
                loss.backward()

        for p in self.model.parameters():
            if p.grad is not None:
                p.grad /= self.train_accumulation_steps

        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad(set_to_none=True)

        return train_loss, metrics

    def train(self):
        """Main training loop."""
        training_time_ms = 0
        best_val_loss = float("inf")
        torch.cuda.synchronize()
        t0 = time.time()

        self.train_loader.reset()
        x, y = self.train_loader.next_batch()

        for step in range(self.args.num_iterations + 1):
            last_step = step == self.args.num_iterations

            # Reset timing after first 10 steps
            if step == 10:
                training_time_ms = 0
                t0 = time.time()

            # Calculate timed steps, accounting for warm-up period
            timed_steps = max(1, step - 10) if step > 10 else 1

            # Validation step
            if last_step or (
                self.args.val_loss_every > 0 and step % self.args.val_loss_every == 0
            ):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.time() - t0)

                val_loss = self.validate()

                if self.is_master:
                    self._log_validation_results(
                        step, val_loss, training_time_ms, timed_steps
                    )
                    if self.args.save_best_model and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_best_model(step, val_loss)

                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                t0 = time.time()

            # Checkpoint saving
            if self.is_master and (
                last_step
                or (self.args.save_every > 0 and step % self.args.save_every == 0)
            ):
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.time() - t0)
                self.save_checkpoint(step)
                torch.cuda.synchronize()
                t0 = time.time()

            if last_step:
                break

            # Training step
            train_loss, metrics = self.train_step(x, y)

            if self.is_master:
                self._log_training_progress(
                    step, train_loss, training_time_ms, t0, timed_steps, metrics
                )

    def _log_validation_results(
        self, step: int, val_loss: float, training_time_ms: float, timed_steps: float
    ):
        """Log validation results."""
        log_msg = f"step:{step}/{self.args.num_iterations} val_loss:{val_loss:.4f} "
        log_msg += f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/timed_steps:.2f}ms"
        print(log_msg)
        with open(self.logfile, "a") as f:
            f.write(log_msg + "\n")

    def _log_training_progress(
        self,
        step: int,
        train_loss: torch.Tensor,
        training_time_ms: float,
        t0: float,
        timed_steps: float,
        metrics: Optional[Dict] = None,
    ):
        """Log training progress."""
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        lr = self.optimizer.param_groups[0]["lr"]

        log_msg = f"step:{step+1}/{self.args.num_iterations} lr:{lr} train_loss:{train_loss.item():.4f} "
        log_msg += (
            f"train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms"
        )

        if metrics:
            log_msg += f" {metrics}"
        print(log_msg)


def main():
    """Entry point for training."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to model config file")
    parser.add_argument("--model_name", type=str, default="bla_gpt", help="Model architecture to use")

    # Get default hyperparameters and add them to argument parser
    hyperparams = Hyperparameters()

    # Add all hyperparameters to the argument parser
    for field_name, field_info in hyperparams.__dataclass_fields__.items():
        field_type = field_info.type
        default_value = getattr(hyperparams, field_name)

        # Get descriptive help message from metadata
        help_msg = field_info.metadata.get("help", f"Configure {field_name}")
        help_text = f"{help_msg} (default: {default_value})"

        # Handle different field types
        if field_type == bool:
            if default_value:
                parser.add_argument(f"--{field_name}", action="store_true", default=default_value,
                                  help=help_text)
            else:
                parser.add_argument(f"--{field_name}", action="store_true", default=default_value,
                                  help=help_text)
        elif field_type == str:
            parser.add_argument(f"--{field_name}", type=str, default=default_value,
                              help=help_text)
        elif field_type == int:
            parser.add_argument(f"--{field_name}", type=int, default=default_value,
                              help=help_text)
        elif field_type == float:
            parser.add_argument(f"--{field_name}", type=float, default=default_value,
                              help=help_text)
        else:
            # For other types, try to infer from default value
            if isinstance(default_value, bool):
                parser.add_argument(f"--{field_name}", action="store_true", default=default_value,
                                  help=help_text)
            elif isinstance(default_value, str):
                parser.add_argument(f"--{field_name}", type=str, default=default_value,
                                  help=help_text)
            elif isinstance(default_value, int):
                parser.add_argument(f"--{field_name}", type=int, default=default_value,
                                  help=help_text)
            elif isinstance(default_value, float):
                parser.add_argument(f"--{field_name}", type=float, default=default_value,
                                  help=help_text)

    args = parser.parse_args()

    # Update hyperparameters with parsed arguments
    for field_name in hyperparams.__dataclass_fields__.keys():
        if hasattr(args, field_name):
            setattr(hyperparams, field_name, getattr(args, field_name))

    trainer = Trainer(hyperparams, args.model_name, args.config)
    trainer.train()


if __name__ == "__main__":
    main()
