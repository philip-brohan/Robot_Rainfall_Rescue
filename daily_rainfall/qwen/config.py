# Default configuration for SFT training
import os
from peft import LoraConfig
from trl import SFTConfig

# Standard image size
# Full size images take too much memory for Qwen3-VL-4B
IMAGE_HEIGHT = 1971
IMAGE_WIDTH = 1200

# Image patch size.
# If not None, images will be split into patches of this size.
PATCH_SIZE = None


def set_SFTConfig(**overrides):
    """
    Build an SFTConfig using the defaults from this module, overridden by provided kwargs.
    Special kwarg: run_id (used to format output_dir and logging_dir).
    """
    run_id = overrides.get("run_id")
    if run_id is None:
        raise ValueError("run_id must be provided to set_SFTConfig")
    pdir = os.getenv("PDIR")
    defaults = {
        "output_dir": f"{pdir}/{run_id}",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "logging_steps": 5,
        "save_strategy": "epoch",
        "learning_rate": 1e-4,
        "bf16": True,
        "max_grad_norm": 0.3,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "constant",
        "push_to_hub": False,
        "report_to": "tensorboard",
        "logging_dir": f"{pdir}/{run_id}/logs",
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }

    # Merge defaults with overrides (overrides take precedence)
    cfg_kwargs = {**defaults, **overrides}
    # remove run_id before passing to SFTConfig
    cfg_kwargs.pop("run_id", None)

    sargs = SFTConfig(**cfg_kwargs)
    sargs.remove_unused_columns = False
    return sargs


# Factory for LoraConfig with overridable defaults
def set_LoraConfig(**overrides):
    """
    Build a LoraConfig using sensible defaults for this project.
    Any provided kwargs override the defaults.
    """
    defaults = {
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "r": 16,
        "bias": "none",
        "target_modules": "all-linear",
        "task_type": "CAUSAL_LM",
        "modules_to_save": ["lm_head", "embed_tokens"],
    }
    cfg_kwargs = {**defaults, **overrides}
    return LoraConfig(**cfg_kwargs)
