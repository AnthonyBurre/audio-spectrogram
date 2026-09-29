import hashlib
import os
import re
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile
from matplotlib.figure import Figure

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_COMPONENT_SEP = re.compile(r"[\s\-_.()\[\]]+")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def _short_stem(audio_file: str) -> str:
    raw = Path(audio_file).stem
    parts = [_NON_ALNUM.sub("", p) for p in _COMPONENT_SEP.split(raw)]
    parts = [p for p in parts if p]
    return "_".join(parts[:2]) or "audio"


def _param_code(params: dict) -> str:
    canonical = "|".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.blake2b(canonical.encode(), digest_size=3).hexdigest()


def _content_code(audio_file: str) -> str:
    # Hash the audio bytes so two different files with the same short_stem
    # don't collide in the cache.
    h = hashlib.blake2b(digest_size=4)
    with open(audio_file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _output_path(audio_file: str, spec_type: str, ext: str, params: dict) -> str:
    return str(
        OUTPUT_DIR / f"{_short_stem(audio_file)}_{spec_type.lower()}"
        f"_{_content_code(audio_file)}_{_param_code(params)}.{ext}"
    )


_DB_REF_MODES = {
    "Per-clip": np.max,
    "Full scale (dBFS)": 1.0,
}


def generate_spectrogram(
    audio_file: str,
    spec_type: str,
    y_scale: str,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    db_ref: str,
    progress=gr.Progress(),
) -> str:
    """Render a dB-scaled STFT or mel spectrogram to PNG and return its path."""
    if audio_file is None:
        raise gr.Error("Upload an audio file first.")

    params = {"n_fft": n_fft, "hop_length": hop_length, "db_ref": db_ref}
    if spec_type == "STFT":
        params["y_scale"] = y_scale
    else:
        params["n_mels"] = n_mels

    output_image_path = _output_path(audio_file, spec_type, "png", params)
    if Path(output_image_path).exists():
        return output_image_path

    progress(None, desc="Loading audio")
    y, sr = librosa.load(audio_file, sr=None)

    progress(None, desc="Computing spectrogram")
    ref = _DB_REF_MODES[db_ref]
    if spec_type == "STFT":
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
        db_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=ref)
        title = "STFT Spectrogram"
        y_axis = y_scale.lower()
    else:
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        db_spectrogram = librosa.power_to_db(mel_spec, ref=ref)
        title = f"Mel Spectrogram — {n_mels} bins"
        y_axis = "mel"

    progress(None, desc="Rendering image")
    # Figure directly (not pyplot) so rendering is safe in Gradio worker threads.
    fig = Figure(figsize=(12, 5))
    ax = fig.subplots()
    img = librosa.display.specshow(
        db_spectrogram,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis=y_axis,
        ax=ax,
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Decibels (dB)")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.savefig(output_image_path, bbox_inches="tight", dpi=150)

    return output_image_path


def reconstruct_audio(
    audio_file: str,
    spec_type: str,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    n_iter: int,
    progress=gr.Progress(),
) -> str:
    """Round-trip audio through a magnitude spectrogram and Griffin-Lim; return the WAV path."""
    if audio_file is None:
        raise gr.Error("Upload an audio file first.")
    if hop_length > n_fft:
        raise gr.Error(
            f"hop_length ({hop_length}) must not exceed n_fft ({n_fft}); "
            "otherwise frames skip samples and can't be overlap-added back."
        )

    params = {"n_fft": n_fft, "hop_length": hop_length, "n_iter": n_iter}
    if spec_type == "Mel":
        # librosa's mel→STFT nnls fails when n_fft//2+1 exceeds ~11× n_mels (empirical).
        min_mels = (n_fft // 2 + 1 + 10) // 11
        if n_mels < min_mels:
            raise gr.Error(
                f"n_fft={n_fft} needs n_mels ≥ {min_mels} for Mel reconstruction "
                f"(got {n_mels}). Increase n_mels or reduce n_fft."
            )
        params["n_mels"] = n_mels

    output_audio_path = _output_path(audio_file, spec_type, "wav", params)
    if Path(output_audio_path).exists():
        return output_audio_path

    progress(None, desc="Loading audio")
    y, sr = librosa.load(audio_file, sr=None)

    if spec_type == "STFT":
        progress(None, desc="Computing STFT magnitude")
        magnitude = np.abs(
            librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
        )
        progress(None, desc=f"Running Griffin-Lim ({n_iter} iters)")
        y_recon = librosa.griffinlim(
            magnitude,
            n_iter=n_iter,
            hop_length=hop_length,
            n_fft=n_fft,
            window="hann",
        )
    else:
        progress(None, desc="Computing mel spectrogram")
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        progress(None, desc=f"Inverting mel + Griffin-Lim ({n_iter} iters)")
        y_recon = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=n_iter,
            window="hann",
        )

    progress(None, desc="Writing WAV")
    soundfile.write(output_audio_path, y_recon, sr)
    return output_audio_path
