from audiomentations import (
    AddBackgroundNoise, AddGaussianNoise, Compose, Gain, OneOf,
    PitchShift, PolarityInversion, TimeStretch, Mp3Compression
)

from torch.utils.data import Dataset
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import librosa
import torch.nn.functional as F
import sentencepiece as spm
from pathlib import Path
import pickle
import hashlib
from typing import List, Dict, Optional, Tuple, Any
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Audio constants will come from config


class AudioCache:
    """Thread-safe audio preprocessing cache"""
    
    def __init__(self, cache_dir: str = "./cache", max_size_gb: float = 5.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024**3
        self.lock = threading.RLock()
        
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{key}.pkl"
    
    def _get_key(self, audio_path: str, sample_rate: int, duration: float, offset: float) -> str:
        """Generate cache key from audio parameters"""
        content = f"{audio_path}_{sample_rate}_{duration}_{offset}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, audio_path: str, sample_rate: int, duration: float, offset: float) -> Optional[np.ndarray]:
        """Get cached audio"""
        with self.lock:
            key = self._get_key(audio_path, sample_rate, duration, offset)
            cache_path = self._get_cache_path(key)
            
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Cache read error: {e}")
                    cache_path.unlink(missing_ok=True)
            return None
    
    def set(self, audio_path: str, sample_rate: int, duration: float, offset: float, audio: np.ndarray):
        """Cache audio data"""
        with self.lock:
            key = self._get_key(audio_path, sample_rate, duration, offset)
            cache_path = self._get_cache_path(key)
            
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(audio, f)
                self._cleanup_cache()
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
    
    def _cleanup_cache(self):
        """Clean up cache if it exceeds size limit"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        if total_size > self.max_size_bytes:
            # Sort by modification time (LRU)
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            
            # Remove oldest files until under limit
            for cache_file in cache_files:
                if total_size <= self.max_size_bytes:
                    break
                total_size -= cache_file.stat().st_size
                cache_file.unlink(missing_ok=True)


class AdvancedAudioAugmentation:
    """Advanced audio augmentation with adaptive strategies"""
    
    def __init__(self, bg_noise_paths: List[str], adaptive: bool = True):
        self.adaptive = adaptive
        self.bg_noise_paths = bg_noise_paths
        
        # Short utterance augmentation (< 3 seconds)
        self.short_augmentation = Compose([
            Gain(min_gain_db=-10, max_gain_db=10, p=0.8),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),
            PitchShift(min_semitones=-1, max_semitones=1, p=0.2),
        ])
        
        # Medium utterance augmentation (3-8 seconds)
        self.medium_augmentation = Compose([
            Gain(min_gain_db=-15, max_gain_db=15, p=0.9),
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
            OneOf([
                AddBackgroundNoise(
                    sounds_path=bg_noise_paths, 
                    min_snr_db=3.0, 
                    max_snr_db=8.0,
                    p=0.7
                ),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.7),
            ], p=0.5)
        ])
        
        # Long utterance augmentation (> 8 seconds)
        self.long_augmentation = Compose([
            Gain(min_gain_db=-20, max_gain_db=10, p=0.95),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),
            PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
            OneOf([
                AddBackgroundNoise(
                    sounds_path=bg_noise_paths,
                    min_snr_db=1.0,
                    max_snr_db=6.0,
                    p=0.8
                ),
                AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.015, p=0.8),
            ], p=0.6),
            OneOf([
                AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.3),
                TimeStretch(min_rate=0.98, max_rate=1.02, p=0.3),
                PitchShift(min_semitones=-1, max_semitones=1, p=0.3),
            ], p=0.3)
        ])
        
        # Noisy environment simulation
        self.noisy_augmentation = Compose([
            Gain(min_gain_db=-25, max_gain_db=5, p=0.95),
            AddBackgroundNoise(
                sounds_path=bg_noise_paths,
                min_snr_db=0.5,
                max_snr_db=4.0,
                p=0.9
            ),
            Mp3Compression(min_bitrate=64, max_bitrate=128, p=0.2),
            TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
        ])
    
    def __call__(self, samples: np.ndarray, sample_rate: int, duration: float) -> np.ndarray:
        """Apply adaptive augmentation based on duration"""
        if not self.adaptive:
            return self.medium_augmentation(samples=samples, sample_rate=sample_rate)
        
        # Apply different augmentation based on duration
        if duration < 3.0:
            return self.short_augmentation(samples=samples, sample_rate=sample_rate)
        elif duration < 8.0:
            # 70% medium, 30% noisy for variety
            if np.random.random() < 0.7:
                return self.medium_augmentation(samples=samples, sample_rate=sample_rate)
            else:
                return self.noisy_augmentation(samples=samples, sample_rate=sample_rate)
        else:
            # 60% long, 40% noisy for robustness
            if np.random.random() < 0.6:
                return self.long_augmentation(samples=samples, sample_rate=sample_rate)
            else:
                return self.noisy_augmentation(samples=samples, sample_rate=sample_rate)


class AudioDataset(Dataset):
    """Audio dataset with automatic train/val split from single metadata file"""
    
    def __init__(self,
                 metadata_file: str,
                 tokenizer_model_path: str,
                 config,
                 mode: str = 'train',  # 'train' or 'val'
                 bg_noise_path: List[str] = None,
                 augment: bool = False,
                 min_text_len: int = None,
                 max_text_len: int = None,
                 enable_caching: bool = True,
                 cache_dir: str = "./cache",
                 num_workers: int = 4,
                 normalize_text: bool = True,
                 adaptive_augmentation: bool = True):
        
        self.config = config
        self.mode = mode
        # Only text length filtering - no duration filtering
        self.min_text_len = min_text_len or config.data.min_text_len
        self.max_text_len = max_text_len or config.data.max_text_len
        self.enable_caching = enable_caching
        self.num_workers = num_workers
        self.normalize_text = normalize_text
        
        # Initialize components
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
        self.device = 'cpu'  # Process on CPU, move to GPU in collate_fn
        
        # Initialize cache
        if enable_caching:
            self.audio_cache = AudioCache(cache_dir)
        else:
            self.audio_cache = None
            
        # Load and split samples automatically
        self.samples = self._load_and_split_samples(metadata_file, mode)
        
        # Initialize augmentation (only for training)
        if augment and bg_noise_path and mode == 'train':
            self.augmentation = AdvancedAudioAugmentation(
                bg_noise_path, 
                adaptive=adaptive_augmentation
            )
        else:
            self.augmentation = None
            
        logger.info(f"✅ {mode.upper()} dataset loaded: {len(self.samples)} samples")
    
    def _load_and_split_samples(self, metadata_file: str, mode: str) -> List[Dict]:
        """Load CSV metadata and split into train/val automatically"""
        logger.info(f"Loading and splitting {metadata_file} for {mode} set...")
        
        try:
            # Read CSV with path|text columns
            df = pd.read_csv(metadata_file, sep='|', names=['path', 'text'], encoding='utf-8')
            logger.info(f"Loaded {len(df)} total rows from {metadata_file}")
            
            # Process all samples first
            all_samples = []
            total_thrown = 0
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing {Path(metadata_file).name}'):
                try:
                    # Convert to dict format
                    sample = {
                        'audio_filepath': row['path'].strip(),
                        'text': row['text'].strip()
                    }
                    
                    # Validate sample
                    if self._is_valid_sample(sample):
                        # Normalize text if enabled
                        if self.normalize_text:
                            sample['text'] = self._normalize_text(sample['text'])
                        all_samples.append(sample)
                    else:
                        total_thrown += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    total_thrown += 1
                    continue
            
            logger.info(f"Valid samples after filtering: {len(all_samples)}, Filtered out: {total_thrown}")
            
            # Shuffle before split if enabled
            if self.config.data.shuffle_before_split:
                np.random.seed(self.config.data.random_seed)
                np.random.shuffle(all_samples)
                logger.info(f"Shuffled data with seed {self.config.data.random_seed}")
            
            # Split into train/val
            train_size = int(len(all_samples) * self.config.data.train_val_split)
            
            if mode == 'train':
                samples = all_samples[:train_size]
                logger.info(f"📊 TRAIN split: {len(samples)} samples ({len(samples)/len(all_samples)*100:.1f}%)")
            elif mode == 'val':
                samples = all_samples[train_size:]
                logger.info(f"📊 VAL split: {len(samples)} samples ({len(samples)/len(all_samples)*100:.1f}%)")
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
            
            return samples
            
        except Exception as e:
            logger.error(f"Error reading {metadata_file}: {e}")
            return []
    
    def _is_valid_sample(self, sample: Dict) -> bool:
        """Check if sample meets filtering criteria (text length only)"""
        try:
            text = sample.get('text', '').strip()
            audio_path = sample.get('audio_filepath', '')
            
            # Check basic requirements
            if not audio_path or not Path(audio_path).exists():
                return False
                
            # Only filter by text length (no duration check)
            if len(text) < self.min_text_len or len(text) > self.max_text_len:
                return False
                
            return True
            
        except Exception:
            return False
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text (basic normalization, can be extended)"""
        # Basic normalization - can be extended with more sophisticated rules
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        return text
    
    def mel_filters(self, device, n_mels: int) -> torch.Tensor:
        """Load mel filterbank matrix"""
        assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
        
        # Try to load from cache first
        cache_path = Path("./weights/mel_filters.npz")
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as f:
                return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
        
        # Fallback to librosa
        import librosa
        mel_basis = librosa.filters.mel(sr=self.config.audio.sample_rate, n_fft=self.config.audio.n_fft, n_mels=n_mels)
        return torch.from_numpy(mel_basis).to(device)
    
    def log_mel_spectrogram(self, audio, n_mels, padding, device):
        """Compute log-Mel spectrogram with optimizations"""
        if device is not None:
            audio = audio.to(device)
            
        if padding > 0:
            audio = F.pad(audio, (0, padding))
            
        window = torch.hann_window(self.config.audio.n_fft).to(audio.device)
        stft = torch.stft(audio, self.config.audio.n_fft, self.config.audio.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        
        filters = self.mel_filters(audio.device, n_mels)
        mel_spec = filters @ magnitudes
        
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
    
    def _load_audio(self, sample: Dict) -> np.ndarray:
        """Load audio with caching (full file, no duration/offset)"""
        audio_path = sample['audio_filepath']
        
        # Try cache first (use file path as key since no duration/offset)
        if self.audio_cache:
            cached_audio = self.audio_cache.get(audio_path, self.config.audio.sample_rate, 0, 0)
            if cached_audio is not None:
                return cached_audio
        
        # Load audio (full file)
        try:
            waveform, _ = librosa.load(
                audio_path,
                sr=self.config.audio.sample_rate
            )
            
            # Cache the result
            if self.audio_cache:
                self.audio_cache.set(audio_path, self.config.audio.sample_rate, 0, 0, waveform)
                
            return waveform
            
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path}: {e}")
            # Return 1 second of silence as fallback
            return np.zeros(int(self.config.audio.sample_rate), dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load audio
            waveform = self._load_audio(sample)
            
            # Apply augmentation
            if self.augmentation is not None:
                # Calculate duration from waveform length
                duration = len(waveform) / self.config.audio.sample_rate
                waveform = self.augmentation(waveform, self.config.audio.sample_rate, duration)
            
            # Tokenize text
            transcript_ids = self.tokenizer.encode_as_ids(sample['text'])
            
            # Convert to tensors
            waveform_tensor = torch.from_numpy(waveform)
            transcript_tensor = torch.tensor(transcript_ids, dtype=torch.long)
            
            # Compute mel spectrogram
            melspec = self.log_mel_spectrogram(waveform_tensor, self.config.audio.n_mels, 0, self.device)
            
            return melspec, transcript_tensor
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            # Return empty sample as fallback
            empty_mel = torch.zeros((self.config.audio.n_mels, 100))  # Minimum length
            empty_transcript = torch.tensor([self.config.model.pad], dtype=torch.long)
            return empty_mel, empty_transcript


def create_collate_fn(config):
    """Factory function to create collate function with config"""
    def collate_fn(batch):
        """Collate function with better padding strategies"""
        mel, text_ids = zip(*batch)
        
        # Filter out empty samples
        valid_samples = [(m, t) for m, t in zip(mel, text_ids) if m.numel() > 0 and t.numel() > 0]
        
        if not valid_samples:
            # Return dummy batch if all samples are invalid
            dummy_mel = torch.zeros((1, config.audio.n_mels, 100))
            dummy_text = torch.tensor([[config.model.pad]], dtype=torch.long)
            dummy_lengths = torch.tensor([100], dtype=torch.int32)
            dummy_text_lengths = torch.tensor([1], dtype=torch.int32)
            return dummy_mel, dummy_lengths, dummy_text, dummy_text_lengths
        
        mel, text_ids = zip(*valid_samples)
        
        # Calculate lengths
        mel_input_lengths = torch.tensor([x.shape[-1] for x in mel], dtype=torch.int32)
        text_input_lengths = torch.tensor([len(x) for x in text_ids], dtype=torch.int32)
        
        # Pad sequences
        max_mel_len = max(x.shape[-1] for x in mel)
        mel_padded = [F.pad(x, (0, max_mel_len - x.shape[-1])) for x in mel]
        
        text_ids_padded = pad_sequence(text_ids, batch_first=True, padding_value=config.model.pad)
        
        return (
            torch.stack(mel_padded),
            mel_input_lengths,
            text_ids_padded,
            text_input_lengths
        )
    
    return collate_fn


# Backward compatibility
def collate_fn(batch):
    """Legacy collate function - loads default config"""
    from config import get_config
    config = get_config()
    return create_collate_fn(config)(batch)


# Factory function for easy dataset creation
def create_dataset(config, mode='train', **kwargs):
    """Factory function to create dataset with automatic train/val split
    
    Args:
        config: ExperimentConfig object
        mode: 'train' or 'val' 
        **kwargs: Additional arguments for AudioDataset
    """
    return AudioDataset(
        metadata_file=config.data.metadata_file,
        tokenizer_model_path=config.model.tokenizer_model_path,
        config=config,
        mode=mode,
        bg_noise_path=config.data.bg_noise_path,
        # No duration parameters - only text length filtering
        **kwargs
    ) 