import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import gradio as gr


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
        y, sr = librosa.load(audio_file)

        if spec_type == "STFT":
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
            db_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            title = f"STFT Spectrogram — {y_scale} Frequency"
            y_axis = y_scale.lower()

        elif spec_type == "Mel":
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

        else:
            raise ValueError("Invalid spectrogram type selected.")

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

        output_image_path = "spectrogram.png"
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
        y, sr = librosa.load(audio_file)

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
        elif spec_type == "Mel":
            # scipy nnls crashes when n_mels is too small relative to n_fft//2+1;
            # empirically the ratio must stay below ~11:1 to stay stable.
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
        else:
            raise ValueError(f"Invalid spectrogram type: {spec_type}")

        output_audio_path = "reconstructed.wav"
        soundfile.write(output_audio_path, y_recon, sr)
        return output_audio_path

    except Exception as e:
        raise gr.Error(e)
