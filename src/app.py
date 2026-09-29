from pathlib import Path

import gradio as gr

from .spectrogram import generate_spectrogram, reconstruct_audio

SPEC_TYPES = ["STFT", "Mel"]
N_FFTS = [512, 1024, 2048, 4096]
Y_SCALES = ["Linear", "Log"]
DB_REFS = ["Per-clip", "Full scale (dBFS)"]

INTRO_MD = """
# Audio Spectrogram Tool

Upload an audio file to view it as a time × frequency map in dB, then **reconstruct audio** from the magnitudes alone.
The round-trip is lossy: phase is discarded and re-estimated with **Griffin-Lim**, and Mel additionally merges frequency bins.
"""


def _toggle_controls(spec_type):
    is_stft = spec_type == "STFT"
    return gr.update(visible=is_stft), gr.update(visible=not is_stft)


def main():
    with gr.Blocks(title="Audio Spectrogram Generator") as demo:
        gr.Markdown(INTRO_MD)

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    type="filepath", label="Audio File (WAV, MP3, FLAC, OGG, …)"
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
                n_fft = gr.Radio(
                    N_FFTS,
                    value=2048,
                    label="FFT Window Size (n_fft)",
                    info="Larger window = finer frequency resolution, coarser time resolution. "
                    "Frequency bin width = sample_rate ÷ n_fft (e.g. 22 Hz at sr=44100, n_fft=2048).",
                )
                hop_length = gr.Slider(
                    128,
                    1024,
                    value=512,
                    step=128,
                    label="Hop Length (hop_length)",
                    info="Smaller step = finer time resolution. Time resolution = hop_length ÷ sample_rate "
                    "(e.g. ~12 ms at sr=44100, hop=512). Overlap = 1 - hop_length / n_fft. "
                    "Must be ≤ n_fft for reconstruction.",
                )
                n_mels = gr.Slider(
                    32,
                    256,
                    value=128,
                    step=32,
                    label="Mel Bins (n_mels)",
                    info="Number of triangular mel filters. More bins = finer perceptual frequency detail. "
                    "For reconstruction, minimum = ceil((n_fft÷2 + 1) ÷ 11)",
                    visible=False,
                )
                db_ref = gr.Radio(
                    DB_REFS,
                    label="dB Reference",
                    value="Per-clip",
                    info="Per-clip: loudest bin = 0 dB, full color range per clip but not comparable across clips. "
                    "Full scale (dBFS): 0 dB = digital full-scale amplitude, comparable across clips but quiet clips look dim.",
                )
                n_iter = gr.Slider(
                    8,
                    64,
                    value=32,
                    step=8,
                    label="Griffin-Lim Iterations",
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
            inputs=[audio_input, spec_type, y_scale, n_fft, hop_length, n_mels, db_ref],
            outputs=spec_output,
        )
        recon_btn.click(
            fn=reconstruct_audio,
            inputs=[audio_input, spec_type, n_fft, hop_length, n_mels, n_iter],
            outputs=audio_output,
        )

    if Path("/.dockerenv").exists():
        # Gradio prints the 0.0.0.0 bind address, which isn't browsable from the host.
        print(
            "Running in Docker: open http://localhost:7860 (or the host port you mapped with -p)"
        )
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
