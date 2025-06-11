from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
from pathlib import Path


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    # Removed max_duration and min_duration - filtering only by text length now


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Encoder config
    n_state: int = 256  # smaller for efficiency
    n_head: int = 4
    n_layer: int = 16
    attention_context_size: Tuple[int, int] = (40, 2)
    
    # Tokenizer config
    vocab_size: int = 1024
    tokenizer_model_path: str = "./weights/tokenizer_spe_bpe_v1024_pad/tokenizer.model"
    
    # CTC specific
    ctc_blank: int = 1024
    rnnt_blank: int = 1024  # Keep for compatibility
    pad: int = 1
    
    # Advanced model features
    dropout: float = 0.1
    label_smoothing: float = 0.1
    use_layer_norm: bool = True

    # Encoder: only FastConformer supported
    encoder_type: str = "fast"
    left_ctx: int = 160
    right_ctx: int = 40

    # FFN hidden dim = n_state * ffn_expansion
    ffn_expansion: int = 4


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training params
    batch_size: int = 16
    num_workers: int = 16
    max_epochs: int = 50
    
    # Optimization
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 2000
    total_steps: int = 3000000
    weight_decay: float = 1e-6
    
    # Training stability
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 50
    num_sanity_val_steps: int = 0  # Disable sanity checking by default
    
    # Mixed precision
    precision: str = "bf16-mixed"
    enable_progress_bar: bool = True

    # Checkpointing
    checkpoint_every_n_steps: int = 1000  # save checkpoints every N training steps
    save_epoch_checkpoint: bool = False   # additionally save checkpoint each epoch end

    # Multi-task learning
    aux_loss_weight: float = 0.2  # weight of auxiliary CTC loss (0 disables)
    lambda_ctc: float = 0.3  # weight for CTC in hybrid CTC+RNNT loss

    # Validation frequency
    val_check_interval: int = 1000  # validate every N training steps


@dataclass
class DataConfig:
    """Data configuration"""
    # Single metadata file - automatically split to train/val
    metadata_file: str = "metadata.csv"
    train_val_split: float = 0.95  # 95% train, 5% val
    bg_noise_path: List[str] = field(default_factory=lambda: ["./datatest/noise/fsdnoisy18k"])
    
    # Data filtering
    min_text_len: int = 1
    max_text_len: int = 60
    
    # Augmentation
    enable_augmentation: bool = True
    augmentation_prob: float = 0.8
    noise_snr_range: Tuple[float, float] = (1.0, 5.0)
    gain_range: Tuple[float, float] = (-25.0, 10.0)
    pitch_shift_range: Tuple[int, int] = (-4, 4)
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    
    # Split options
    shuffle_before_split: bool = True
    random_seed: int = 42


@dataclass
class InferenceConfig:
    """Inference configuration"""
    beam_size: int = 5
    use_beam_search: bool = False
    length_penalty: float = 0.3
    use_language_model: bool = False
    lm_weight: float = 0.5


@dataclass
class PathConfig:
    """Path configuration"""
    # Model weights
    pretrained_encoder_weight: str = "./weights/phowhisper_small_encoder.pt"
    tokenizer_model_path: str = "./weights/tokenizer_spe_bpe_v1024_pad/tokenizer.model"
    
    # Logging and checkpoints
    log_dir: str = "./checkpoints"
    checkpoint_dir: str = "./checkpoints"
    tensorboard_dir: str = "./logs/tensorboard"
    wandb_project: Optional[str] = None
    
    # Data paths
    dataset_dir: str = "./dataset"
    weights_dir: str = "./weights"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for path in [self.log_dir, self.checkpoint_dir, self.tensorboard_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str = "improved_ctc_whisper"
    description: str = "Improved CTC-based ASR with PhoWhisper encoder"
    version: str = "1.0"
    
    # Sub-configs
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ExperimentConfig':
        """Create config from dictionary"""
        # Handle nested dataclasses properly
        kwargs = {}
        for key, value in config_dict.items():
            if key == 'audio' and isinstance(value, dict):
                kwargs[key] = AudioConfig(**value)
            elif key == 'model' and isinstance(value, dict):
                # Handle tuple conversion for attention_context_size
                if 'attention_context_size' in value and isinstance(value['attention_context_size'], list):
                    value = value.copy()
                    value['attention_context_size'] = tuple(value['attention_context_size'])
                kwargs[key] = ModelConfig(**value)
            elif key == 'training' and isinstance(value, dict):
                kwargs[key] = TrainingConfig(**value)
            elif key == 'data' and isinstance(value, dict):
                # Handle tuple conversions
                value_copy = value.copy()
                for field_name, field_value in value_copy.items():
                    if field_name.endswith('_range') and isinstance(field_value, list) and len(field_value) == 2:
                        value_copy[field_name] = tuple(field_value)
                kwargs[key] = DataConfig(**value_copy)
            elif key == 'inference' and isinstance(value, dict):
                kwargs[key] = InferenceConfig(**value)
            elif key == 'paths' and isinstance(value, dict):
                kwargs[key] = PathConfig(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)
    
    def save(self, path: str):
        """Save config to JSON file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()


def get_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """Get configuration from file or default"""
    if config_path and os.path.exists(config_path):
        return ExperimentConfig.load(config_path)
    return DEFAULT_CONFIG


# update_constants_from_config function removed - we now use config system exclusively


if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig()
    print("Default configuration:")
    print(f"Model: {config.model.n_state}-{config.model.n_head}-{config.model.n_layer}")
    print(f"Training: LR={config.training.learning_rate}, Batch={config.training.batch_size}")
    print(f"Audio: SR={config.audio.sample_rate}, MELs={config.audio.n_mels}")
    
    # Save config
    config.save("config_example.json")
    print("Config saved to config_example.json") 