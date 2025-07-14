import librosa
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

def generate_spectrogram(audio_file, spec_type, n_fft, hop_length):
    """
    Generates a spectrogram with selectable types and parameters.

    Args:
        audio_file (str): The file path of the input audio file.
        spec_type (str): Type of spectrogram ('Linear Magnitude', 'Linear Power', 'Logarithmic', 'Mel').
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between successive frames.

    Returns:
        str: The file path of the generated spectrogram image.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file)

        # --- Spectrogram Calculation ---
        if spec_type == 'Linear Magnitude' or spec_type == 'Linear Power':
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
            if spec_type == 'Linear Magnitude':
                spectrogram = np.abs(stft)
                title = 'Linear Magnitude Spectrogram'
                y_axis = 'linear'
            else: # Linear Power
                spectrogram = np.abs(stft)**2
                title = 'Linear Power Spectrogram'
                y_axis = 'linear'
            db_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        elif spec_type == 'Logarithmic':
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            db_spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
            title = 'Logarithmic Spectrogram (dB)'
            y_axis = 'log'

        elif spec_type == 'Mel':
            # Use power spectrogram as the basis for Mel
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, power=2.0
            )
            db_spectrogram = librosa.power_to_db(mel_spec, ref=np.max)
            title = 'Mel Spectrogram (dB)'
            y_axis = 'mel'

        else:
            raise ValueError("Invalid spectrogram type selected.")


        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(12, 5))
        img = librosa.display.specshow(
            db_spectrogram,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis=y_axis,
            ax=ax
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Decibels')
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')

        output_image_path = "spectrogram.png"
        plt.savefig(output_image_path, bbox_inches='tight')
        plt.close(fig)

        return output_image_path

    except Exception as e:
        raise gr.Error(e)