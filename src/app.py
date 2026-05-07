import gradio as gr

from .spectrogram import generate_spectrogram, reconstruct_audio

SPEC_TYPES = ["STFT", "Mel"]
Y_SCALES = ["Linear", "Log"]

INTRO_MD = """
# Audio Spectrogram Tool

This tool transforms an audio signal into a time x frequency map of amplitudes in decibels (dB), showing how a sound's frequency content evolves moment to moment.

Upload an audio file to visualize it as a spectrogram, then **reconstruct audio** from the magnitude values alone. This round-trip is lossy: phase information is thrown away when computing the spectrogram and must be estimated back using the iterative **Griffin-Lim algorithm**. 
Increase iterations to recover phase more faithfully, at the cost of compute time. 
Mel spectrograms carry an additional loss because the mel filterbank compresses many frequency bins into fewer perceptual ones before inversion.
"""


def _toggle_controls(spec_type):
    is_stft = spec_type == "STFT"
    return gr.update(visible=is_stft), gr.update(visible=not is_stft)


def main():
    """Defines and launches the Gradio web interface."""

    with gr.Blocks(title="Audio Spectrogram Generator") as demo:
        gr.Markdown(INTRO_MD)

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    type="filepath", label="Audio File (MP3, WAV, FLAC)"
                )
                spec_type = gr.Radio(
                    SPEC_TYPES,
                    label="Spectrogram Type",
                    value="STFT",
                    info="Controls how frequency bins are scaled and grouped. All types display amplitude in dB. "
                    "STFT favored for harmonic structure, overtones, and pitch. "
                    "Mel favored for speech intelligibility and music ML features.",
                )
                y_scale = gr.Radio(
                    Y_SCALES,
                    label="Frequency Axis Scale",
                    value="Log",
                    info="Linear: uniform Hz spacing, good for seeing overtone series. "
                    "Log: octave-spaced, matches human pitch perception.",
                )
                n_fft = gr.Slider(
                    512,
                    4096,
                    value=2048,
                    step=512,
                    label="n_fft — FFT window size",
                    info="Larger window = finer frequency resolution, coarser time resolution. "
                    "Frequency bin width = sample_rate ÷ n_fft (e.g. 22 Hz at sr=44100, n_fft=2048).",
                )
                hop_length = gr.Slider(
                    128,
                    1024,
                    value=512,
                    step=128,
                    label="hop_length — frame step (samples)",
                    info="Smaller step = finer time resolution. Time resolution = hop_length ÷ sample_rate "
                    "(e.g. ~12 ms at sr=44100, hop=512). Overlap = 1 − hop_length / n_fft.",
                )
                n_mels = gr.Slider(
                    64,
                    256,
                    value=128,
                    step=32,
                    label="n_mels — mel filter banks",
                    info="Number of triangular mel filters. More bins = finer perceptual frequency detail. "
                    "Only applies to the Mel spectrogram. Minimum 64 for audio reconstruction.",
                    visible=False,
                )
                n_iter = gr.Slider(
                    8,
                    64,
                    value=32,
                    step=8,
                    label="Griffin-Lim iterations",
                    info="Phase is discarded when computing a spectrogram. Griffin-Lim estimates it back "
                    "iteratively. More iterations = closer reconstruction, slower compute.",
                )

                with gr.Row():
                    spec_btn = gr.Button("Generate Spectrogram", variant="primary")
                    recon_btn = gr.Button("Reconstruct Audio")

            with gr.Column():
                spec_output = gr.Image(type="filepath", label="Spectrogram")
                audio_output = gr.Audio(type="filepath", label="Reconstructed Audio")

        spec_type.change(
            fn=_toggle_controls, inputs=spec_type, outputs=[y_scale, n_mels]
        )

        spec_btn.click(
            fn=generate_spectrogram,
            inputs=[audio_input, spec_type, y_scale, n_fft, hop_length, n_mels],
            outputs=spec_output,
        )
        recon_btn.click(
            fn=reconstruct_audio,
            inputs=[audio_input, spec_type, n_fft, hop_length, n_mels, n_iter],
            outputs=audio_output,
        )

    print("---------------------------------------------------------------------")
    print("If running in a Docker container, access app at: http://localhost:7860")
    print("---------------------------------------------------------------------")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
