import argparse
from pathlib import Path
import torch
import torchaudio
import numpy as np
from loguru import logger

from config import ExperimentConfig, get_config
from models.fast_conformer import FastConformerEncoder
from models.rnnt_decoder import RNNTDecoder
from models.rnnt_streaming import StreamingGreedyRNNT
import sentencepiece as spm


def load_model(cfg: ExperimentConfig, ckpt_path: str, device: str):
    # init encoder
    encoder = FastConformerEncoder(
        n_mels=cfg.audio.n_mels,
        d_model=cfg.model.n_state,
        n_heads=cfg.model.n_head,
        n_layers=cfg.model.n_layer,
        left_ctx=cfg.model.left_ctx,
        right_ctx=cfg.model.right_ctx,
        dropout=0.0,
        ffn_expansion=cfg.model.ffn_expansion,
    ).to(device).eval()

    state = torch.load(ckpt_path, map_location=device)
    enc_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state, strict=False)

    # RNN-T decoder
    rnnt_decoder = RNNTDecoder(
        vocab_size=cfg.model.vocab_size,
        enc_dim=cfg.model.n_state,
    ).to(device).eval()
    rnnt_state = {k.replace("rnnt_decoder.", ""): v for k, v in state.items() if k.startswith("rnnt_decoder.")}
    rnnt_decoder.load_state_dict(rnnt_state, strict=False)

    # Streaming greedy decoder
    streaming_decoder = StreamingGreedyRNNT(rnnt_decoder, device=device)
    return encoder, rnnt_decoder, streaming_decoder


def log_mel(audio, cfg: ExperimentConfig):
    with torch.no_grad():
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.audio.sample_rate,
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            n_mels=cfg.audio.n_mels,
        )(audio)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = (mel + 4.0) / 4.0
    return mel


def stream_transcribe(wav_path: str, cfg: ExperimentConfig, encoder, rnnt_decoder, streaming_decoder, tokenizer, device):
    wav, sr = torchaudio.load(wav_path)
    wav = torchaudio.functional.resample(wav, sr, cfg.audio.sample_rate)
    wav = wav.squeeze(0).to(device)

    chunk_len = int(cfg.audio.sample_rate * 0.64)  # 640 ms
    stride = int(cfg.audio.sample_rate * 0.48)      # 160 ms overlap

    offset = 0
    cache = encoder.init_cache(batch_size=1, device=device)
    streaming_decoder.reset()  # Reset streaming decoder state
    collected = []

    while offset < wav.numel():
        end = min(offset + chunk_len, wav.numel())
        chunk = wav[offset:end]
        mel = log_mel(chunk, cfg).unsqueeze(0)  # (1,n_mels,T)
        with torch.no_grad():
            enc, cache = encoder.stream_step(mel, cache)
            # Use streaming RNN-T decoder
            pred_ids = streaming_decoder.infer(enc)
            collected.extend(pred_ids)
        offset += stride

    return tokenizer.decode(collected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="config json")
    args = parser.parse_args()

    cfg = get_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder, rnnt_decoder, streaming_decoder = load_model(cfg, args.checkpoint, device)
    tokenizer = spm.SentencePieceProcessor(model_file=cfg.model.tokenizer_model_path)

    text = stream_transcribe(args.audio, cfg, encoder, rnnt_decoder, streaming_decoder, tokenizer, device)
    print(">>", text)


if __name__ == "__main__":
    main() 