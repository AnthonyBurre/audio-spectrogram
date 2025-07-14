import gradio as gr
import os

# Import the core function from our spectrogram module
from .spectrogram import generate_spectrogram

def main():
    """Defines and launches the Gradio web interface."""

    # Define the Gradio interface
    demo = gr.Interface(
        fn=generate_spectrogram,
        inputs=[
            gr.Audio(type="filepath", label="Upload Audio File (MP3, WAV, FLAC)"),
            gr.Radio(
                ['Linear Magnitude', 'Linear Power', 'Logarithmic', 'Mel'],
                label="Spectrogram Type",
                value='Logarithmic' # Default value
            ),
            gr.Slider(512, 4096, value=2048, step=512, label="n_fft", info="FFT window size."),
            gr.Slider(128, 1024, value=512, step=128, label="hop_length", info="Number of samples between frames."),
        ],
        outputs=gr.Image(type="filepath", label="Generated Spectrogram"),
        title="Audio Spectrogram Generator 🎼",
        description="Upload an audio file to generate its spectrogram.",
        flagging_mode="never",
    )
    
    # Launch the web interface
    print("---------------------------------------------------------------------")
    print(f"If running in a Docker container, access app at: http://localhost:7860")
    print("---------------------------------------------------------------------")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860
    )

# This makes the script runnable
if __name__ == "__main__":
    main()
