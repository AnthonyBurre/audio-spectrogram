# Audio File <-> Spectrogram Conversion

A spectrogram is a `time × frequency` heatmap showing how the content of a signal evolves moment to moment. This project is a tool for visualizing audio files as spectrograms, and reconstructing them via Griffin-Lim.

```bash
pip install -r requirements.txt
python -m src.app
# → http://localhost:7860 or http://0.0.0.0:7860
```

Or with Docker:

```bash
docker build -t audio-spectrogram .
docker run --rm -p 7860:7860 -v "$(pwd)/outputs:/app/outputs" audio-spectrogram
# → http://localhost:7860
```

Generated spectrograms and reconstructed audio are written to `outputs/` which doubles as a cache.

Accepts any format librosa can read (`wav`, `mp3`, `flac`, `ogg`, `m4a`, …). The input's sample rate is preserved but stereo files are downmixed to mono.

---

## Example

Demos of round-trip on a 60-second music clip ([717x-chillwave.mp3](https://github.com/user-attachments/files/28295068/717x-chillwave.mp3)):

**STFT reconstruction**

[717x-chillwave-reconstructed-stft.mp4](https://github.com/user-attachments/assets/9dced0ed-0e54-4cb2-b447-439e72758ea0)

`n_fft=2048`, `hop_length=512`, 32 Griffin-Lim iterations: loses some quality but still gets all the ideas through.

**Mel reconstruction**

[717x-chillwave-reconstructed-mel.mp4](https://github.com/user-attachments/assets/e0628275-527e-4183-83c6-6f479ad7146a)

`n_fft=512`, `hop_length=128`, `n_mels=32`, 64 Griffin-Lim iterations: very muffled and degraded but recognizable.

---

## Spectrogram Types

### STFT

The Short-Time Fourier Transform divides the signal into overlapping time frames, applies a Hann window to each, and computes each frame's FFT. Without windowing, the hard cut at each frame boundary leaks a sinusoid's energy across the whole spectrum (spectral leakage); the Hann taper confines it to a few neighboring bins.

The result is a complex matrix `S[k, t]`, where `k` indexes frequency bins (`0` through `n_fft / 2`) and `t` indexes time frames. The spectrogram discards phase and plots amplitudes `|S[k, t]|` in dB against a reference amplitude `A_ref`:

```
dB = 20 · log₁₀(|S| / A_ref)
```

### Mel

A mel spectrogram re-bins the STFT's linearly-spaced FFT bins onto a coarser, perceptually-motivated frequency axis, with dense spacing at low frequencies where human pitch resolution is finest. That makes it a compact feature for ML.

The input is the power spectrogram (`|S|²`), because acoustic power is proportional to the square of pressure, and summing bin powers within a filter is physically meaningful in a way that summing amplitudes is not.

Plotted in dB against a reference power `P_ref`:

```
dB = 10 · log₁₀(P / P_ref)
```

where `P = |S[k, t]|²`.

Each of the `n_mels` filters is a triangle in the frequency domain: it ramps up from zero, peaks at its center frequency, ramps back down, and overlaps its neighbors so no FFT bin is left unweighted.

The dot product of one filter with one frame's power spectrum yields one mel bin: the total power in that perceptual band at that moment. Across all filters and frames, this produces an `(n_mels × time_frames)` matrix. Detail within each band is lost, so mel reconstruction sounds more degraded than STFT reconstruction.

---

## Reconstruction

Each STFT bin `S[k, t]` is a complex number `a + bi`. From it you can derive two quantities: **magnitude** `√(a² + b²)` and **phase** `atan2(b, a)` (where in its oscillation cycle that component sits). The spectrogram only displays magnitudes, but we need both for the iSTFT (inverse STFT) inversion process.

**Griffin-Lim** estimates a plausible phase by iterating between the time and frequency domains:

1. Start with a random phase estimate.
2. **Apply the magnitude constraint.** Combine the known magnitudes with the current phase estimate to form a complex matrix.
3. **Invert to the time domain (iSTFT).** For each frame, run an inverse FFT to produce a windowed time slice, then overlap-add successive slices to reconstruct a continuous waveform. Because the current phase estimate isn't yet self-consistent across frames, adjacent slices disagree where they overlap and the overlap-add averages out the disagreement.
4. Re-compute the STFT to get a new phase estimate.
5. Repeat from step 2 for the number of passes set by `Griffin-Lim Iterations`.

The waveform from the final iSTFT pass is the reconstructed audio. Each iteration drives the phase toward self-consistency, but never recovers the original (which was discarded).

---

## Parameters

### `FFT Window Size (n_fft)` *(display + reconstruction)*

The number of samples analyzed by a single FFT. Controls the fundamental **time–frequency resolution tradeoff**:

- **Frequency resolution:** `Δf = sample_rate / n_fft` Hz per bin
- **Window duration:** `n_fft / sample_rate` seconds

| Larger `n_fft` | Smaller `n_fft` |
|---|---|
| Finer frequency bins | Coarser frequency bins |
| Wider time window, blurs fast transients | Narrower window, captures sharp attacks |

Options are 512, 1024, 2048, 4096. Use 512 for drums and transients, 2048 for general use, and 4096 for low-frequency detail.

### `Hop Length (hop_length)` *(display + reconstruction)*

The number of samples the window advances between frames. Sets time resolution along the spectrogram x-axis:

```
Δt = hop_length / sample_rate   (e.g. 512 / 44100 ≈ 11.6 ms)
```

The defaults (`n_fft=2048`, `hop_length=512`) give 75% overlap. Hann windows overlap-add cleanly at 50% overlap or more (`hop_length ≤ n_fft / 2`); higher overlap gives Griffin-Lim more redundancy to converge on a consistent phase. Reconstruction requires `hop_length ≤ n_fft`, or frames skip samples entirely.

### `Mel Bins (n_mels)` *(display + reconstruction)*

The number of triangular filters in the mel filterbank (32–256). More filters preserve finer perceptual frequency resolution at the cost of a larger feature matrix. The default of 128 is standard for music; speech models often use 40–80.

For reconstruction, `n_mels` must be large enough relative to `n_fft` to keep the mel→STFT inversion stable (minimum = `ceil((n_fft // 2 + 1) / 11)`).

> Hidden when STFT is selected

### `Griffin-Lim Iterations` *(reconstruction only)*

Number of passes to run through the iterative reconstruction process described above. More iterations reduce phase inconsistency and improve perceptual quality, with diminishing returns.

### `Frequency Axis Scale` *(display only)*

Determines how the y-axis is rendered:

- **Linear**: uniform Hz spacing from `0` to `sample_rate / 2` (Nyquist). Harmonic overtones appear as evenly-spaced horizontal bands.
- **Log**: logarithmic spacing so each octave (doubling of frequency) occupies equal vertical height. Matches human pitch perception; melodic intervals are easier to identify.

> Hidden when Mel is selected; the mel axis is always mel-scaled

### `dB Reference` *(display only)*

Sets which value maps to 0 dB on the color scale.

- **Per-clip** (default): the loudest bin in the clip is the reference. Max is always 0 dB; everything else is negative. Each clip uses the full color range, but values are not comparable across clips.
- **Full scale (dBFS)**: 0 dB = digital full-scale amplitude (1.0). Values are comparable across clips, but quiet recordings render dim because their peak sits well below 0.
