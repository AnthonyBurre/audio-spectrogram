import hashlib
import os
import re
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import gradio as gr

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))

_COMPONENT_SEP = re.compile(r"[\s\-_.()\[\]]+")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def _short_stem(audio_file: str) -> str:
    raw = Path(audio_file).stem
    parts = [_NON_ALNUM.sub("", p) for p in _COMPONENT_SEP.split(raw)]
    parts = [p for p in parts if p]
    return "_".join(parts[:2]) or "audio"


def _param_code(params: dict) -> str:
    canonical = "|".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.md5(canonical.encode()).hexdigest()[:6]


def _output_path(audio_file: str, spec_type: str, ext: str, params: dict) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(
        OUTPUT_DIR
        / f"{_short_stem(audio_file)}_{spec_type.lower()}_{_param_code(params)}.{ext}"
    )


def generate_spectrogram(audio_file, spec_type, y_scale, n_fft, hop_length, n_mels):
    """
    Generates a spectrogram with selectable types and parameters.

    Args:
        audio_file (str): The file path of the input audio file.
        spec_type (str): Type of spectrogram ('STFT', 'Mel').
        y_scale (str): Frequency axis scale for STFT ('Linear', 'Log'). Ignored for Mel.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.
        n_mels (int): Number of mel filter banks (Mel type only).

    Returns:
        str: The file path of the generated spectrogram image.
    """
    try:
        params = {"n_fft": n_fft, "hop_length": hop_length}
        if spec_type == "STFT":
            params["y_scale"] = y_scale
        elif spec_type == "Mel":
            params["n_mels"] = n_mels
        else:
            raise ValueError("Invalid spectrogram type selected.")

        output_image_path = _output_path(audio_file, spec_type, "png", params)
        if Path(output_image_path).exists():
            return output_image_path

        y, sr = librosa.load(audio_file, sr=None)

        if spec_type == "STFT":
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
            db_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
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
            db_spectrogram = librosa.power_to_db(mel_spec, ref=np.max)
            title = f"Mel Spectrogram — {n_mels} bins"
            y_axis = "mel"

        fig, ax = plt.subplots(figsize=(12, 5))
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

        plt.savefig(output_image_path, bbox_inches="tight")
        plt.close(fig)

        return output_image_path

    except Exception as e:
        raise gr.Error(e)


def reconstruct_audio(audio_file, spec_type, n_fft, hop_length, n_mels, n_iter):
    """
    Round-trip an audio file through a spectrogram and back to audio.

    Inversion is lossy: phase is discarded and recovered via Griffin-Lim.
    Mel spectrograms additionally collapse spectral detail into mel bins.

    Args:
        audio_file (str): The file path of the input audio file.
        spec_type (str): Type of spectrogram ('STFT', 'Mel').
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.
        n_mels (int): Number of mel filter banks (Mel type only).
        n_iter (int): Number of Griffin-Lim iterations.

    Returns:
        str: The file path of the reconstructed audio (.wav).
    """
    try:
        params = {"n_fft": n_fft, "hop_length": hop_length, "n_iter": n_iter}
        if spec_type == "Mel":
            params["n_mels"] = n_mels
        elif spec_type != "STFT":
            raise ValueError(f"Invalid spectrogram type: {spec_type}")

        output_audio_path = _output_path(audio_file, spec_type, "wav", params)
        if Path(output_audio_path).exists():
            return output_audio_path

        y, sr = librosa.load(audio_file, sr=None)

        if spec_type == "STFT":
            magnitude = np.abs(
                librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
            )
            y_recon = librosa.griffinlim(
                magnitude,
                n_iter=n_iter,
                hop_length=hop_length,
                n_fft=n_fft,
                window="hann",
            )
        else:
            # Numerical-stability bound on the pseudoinverse used by mel_to_audio:
            # scipy's nnls fails when n_mels is too small relative to n_fft//2+1.
            # This is an implementation constraint, not a mathematical requirement
            # of mel inversion itself. Empirically the ratio must stay below ~11:1.
            min_mels = (n_fft // 2 + 1 + 10) // 11
            if n_mels < min_mels:
                raise ValueError(
                    f"With n_fft={n_fft}, Mel reconstruction requires n_mels ≥ {min_mels} "
                    f"(you selected {n_mels}). The mel→STFT inversion (via scipy nnls) "
                    f"crashes when n_mels is too small relative to the FFT size. "
                    f"Increase n_mels or reduce n_fft."
                )
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            )
            y_recon = librosa.feature.inverse.mel_to_audio(
                mel_spec,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_iter=n_iter,
                window="hann",
            )

        soundfile.write(output_audio_path, y_recon, sr)
        return output_audio_path

    except Exception as e:
        raise gr.Error(e)
