import torch
import torch.nn.functional as F
import torchaudio
import sentencepiece as spm
from loguru import logger
import librosa
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import time
import concurrent.futures
from dataclasses import dataclass
import json
from tqdm import tqdm

# Import model components
from models.fast_conformer import FastConformerEncoder
from models.rnnt_decoder import RNNTDecoder
from models.rnnt_streaming import StreamingGreedyRNNT
from config import (
    ExperimentConfig,
    AudioConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    InferenceConfig,
    PathConfig,
    get_config
)

# Add safe globals for checkpoint loading
torch.serialization.add_safe_globals([
    ExperimentConfig,
    AudioConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    InferenceConfig,
    PathConfig
])


@dataclass
class InferenceResult:
    """Container for inference results"""
    file_path: str
    transcription: str
    confidence_score: float
    processing_time: float
    method: str  # 'greedy' or 'beam_search'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': self.file_path,
            'transcription': self.transcription,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'method': self.method
        }


class RNNTInference:
    """Advanced RNN-T inference with streaming capabilities"""

    def __init__(self, checkpoint_path: str, config: Optional[ExperimentConfig] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or get_config()
        logger.info(f"🚀 Initializing RNN-T inference on {self.device}")

        self._load_model(checkpoint_path)
        self._init_tokenizer()
        self._init_decoder()

        logger.info("✅ Inference engine ready!")
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        logger.info(f"📦 Loading model from {checkpoint_path}")

        # Initialize Fast Conformer encoder
        self.encoder = FastConformerEncoder(
            n_mels=self.config.audio.n_mels,
            d_model=self.config.model.n_state,
            n_heads=self.config.model.n_head,
            n_layers=self.config.model.n_layer,
            dropout=self.config.model.dropout,
            ffn_expansion=self.config.model.ffn_expansion,
            left_ctx=self.config.model.left_ctx,
            right_ctx=self.config.model.right_ctx,
        )

        # Initialize RNN-T decoder
        self.rnnt_decoder = RNNTDecoder(
            vocab_size=self.config.model.vocab_size,
            enc_dim=self.config.model.n_state,
        )

        # Load checkpoint with weights_only=True for security
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # Handle different checkpoint formats
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Separate encoder and RNN-T decoder weights
        encoder_weights = {}
        rnnt_weights = {}

        for key, value in state_dict.items():
            if 'alibi' in key:  # Skip ALiBi weights
                continue
            elif key.startswith('encoder.'):
                encoder_weights[key.replace('encoder.', '')] = value
            elif key.startswith('rnnt_decoder.'):
                rnnt_weights[key.replace('rnnt_decoder.', '')] = value

        # Load weights
        self.encoder.load_state_dict(encoder_weights, strict=False)
        self.rnnt_decoder.load_state_dict(rnnt_weights, strict=False)

        # Move to device and set to eval mode
        self.encoder = self.encoder.to(self.device).eval()
        self.rnnt_decoder = self.rnnt_decoder.to(self.device).eval()
        
    def _init_tokenizer(self):
        """Initialize SentencePiece tokenizer"""
        self.tokenizer = spm.SentencePieceProcessor(model_file=self.config.model.tokenizer_model_path)
        
    def _init_decoder(self):
        """Initialize RNN-T streaming decoder"""
        self.streaming_decoder = StreamingGreedyRNNT(self.rnnt_decoder, device=self.device)
    
    def log_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute log mel spectrogram"""
        window = torch.hann_window(self.config.audio.n_fft).to(audio.device)
        stft = torch.stft(audio, self.config.audio.n_fft, self.config.audio.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        
        # Use librosa mel filters
        mel_basis = librosa.filters.mel(sr=self.config.audio.sample_rate, n_fft=self.config.audio.n_fft, n_mels=self.config.audio.n_mels)
        mel_basis = torch.from_numpy(mel_basis).to(audio.device)
        
        mel_spec = torch.matmul(mel_basis, magnitudes)
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = (log_spec + 4.0) / 4.0
        
        return log_spec
    
    def transcribe_single(self, audio_path: str, use_streaming: bool = True) -> InferenceResult:
        """Transcribe single audio file using RNN-T"""
        start_time = time.time()

        try:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=self.config.audio.sample_rate)
            audio_tensor = torch.from_numpy(audio).to(self.device)

            with torch.no_grad():
                # Compute features
                mels = self.log_mel_spectrogram(audio_tensor)
                x = mels.unsqueeze(0)  # Add batch dimension
                x_len = torch.tensor([x.shape[2]]).to(self.device)

                # Forward pass through encoder
                enc_out, enc_len, _ = self.encoder(x, x_len, return_intermediate=False)

                # Decode using streaming RNN-T decoder
                if use_streaming:
                    self.streaming_decoder.reset()
                    decoded_tokens = self.streaming_decoder.infer(enc_out)
                    method = "streaming_greedy"
                else:
                    # Fallback to simple greedy decoding (not implemented here)
                    decoded_tokens = []
                    method = "greedy"

                # Get transcription
                transcription = self.tokenizer.decode(decoded_tokens) if decoded_tokens else ""
                confidence = 0.8  # Placeholder confidence score

        except Exception as e:
            logger.error(f"❌ Error processing {audio_path}: {e}")
            transcription = ""
            confidence = 0.0
            method = "error"

        processing_time = time.time() - start_time

        return InferenceResult(
            file_path=audio_path,
            transcription=transcription,
            confidence_score=confidence,
            processing_time=processing_time,
            method=method
        )


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RNN-T ASR Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming decoding")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")
    parser.add_argument("--config", type=str, help="Path to config file (optional)")
    
    args = parser.parse_args()
    
    # Resolve configuration: explicit path -> alongside checkpoint -> default
    from pathlib import Path
    config: ExperimentConfig

    if args.config:
        # User-specified config file
        config = ExperimentConfig.load(args.config)
        logger.info(f"📝 Loaded config from {args.config}")
    else:
        # Try to find config.json next to checkpoint
        ckpt_dir = Path(args.checkpoint).expanduser().resolve().parent
        candidate = ckpt_dir / "config.json"
        if candidate.exists():
            config = ExperimentConfig.load(str(candidate))
            logger.info(f"📝 Loaded config from {candidate}")
        else:
            config = get_config()
            logger.warning("⚠️ Config file not provided and none found next to checkpoint. Using default config – ensure compatibility!")

    logger.info(f"📊 Model vocab size: {config.model.vocab_size}")
    
    # Initialize inference
    inference = RNNTInference(args.checkpoint, config, args.device)

    # Transcribe
    result = inference.transcribe_single(args.audio, args.streaming)
    
    print(f"🎯 Transcription: {result.transcription}")
    print(f"⏱️ Time: {result.processing_time:.2f}s")
    print(f"📈 Confidence: {result.confidence_score:.3f}")


if __name__ == "__main__":
    main() 