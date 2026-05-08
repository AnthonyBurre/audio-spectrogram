# Audio File <-> Spectrogram Conversion

A spectrogram is a time × frequency map showing how a signal's content evolves moment to moment. This project is a tool for visualizing audio files as spectrograms, and reconstructing them via Griffin-Lim.

```bash
pip install -r requirements.txt
python -m src.app
# → http://localhost:7860
```

Or with Docker:

```bash
docker build -t audio-spectrogram .
docker run --rm -p 7860:7860 -v "$(pwd)/outputs:/app/outputs" audio-spectrogram
# → http://localhost:7860
```

Generated spectrograms and reconstructed audio are written to `outputs/` which doubles as a cache.

---

## Spectrogram types

All types display values on a decibel (dB) scale, which is fundamentally a power ratio: `dB = 10 · log₁₀(P / P_ref)`. Because `P ∝ A²`, for amplitude inputs the equivalent is `20 · log₁₀(A / A_ref)`

dB has no absolute meaning without a reference. Common references include full-scale digital amplitude (dBFS) and sound pressure level (dB SPL). This project uses the loudest bin in the clip itself as the reference, so the maximum is always 0 dB and everything else is negative. Values reflect dynamics within a clip but are not comparable across clips.

### STFT

The Short-Time Fourier Transform divides the signal into overlapping time frames, Hann windows each frame to mitigate spectral leakage, and computes each FFT. Without windowing, the hard cut-off at each frame boundary introduces artificial discontinuities that leak energy across frequency bins (spectral leakage). The Hann window smooths those edges so energy from a single sinusoid still spreads across a few bins, but not far across the spectrum.

The result is a complex matrix `S[k, t]`, where `k` indexes frequency bins (`0` through `n_fft / 2`) and `t` indexes time frames. The spectrogram discards phase and plots amplitudes `|S[k, t]|` in dB, normalized to the loudest bin:

```
dB = 20 · log₁₀(|S| / max|S|)
```

### Mel

A mel spectrogram re-bins the STFT's many linearly-spaced FFT bins onto a coarser, perceptually-motivated frequency axis. The output is much smaller — `n_mels` is typically 40–128, versus `n_fft/2 + 1` ≈ 1025 for `n_fft=2048` — with dense spacing at low frequencies, where human pitch resolution is finest. That makes it a compact feature for ML, and a lossy target for reconstruction.

The input is the **power spectrogram**: acoustic power is proportional to the square of pressure, and powers from incoherent sources add, so summing FFT-bin powers within each mel filter is physically meaningful in a way that summing amplitudes is not.

Plotted in dB, normalized to the loudest bin:

```
dB = 10 · log₁₀(P / max P)
```

where `P = |S[k, t]|²`. Strictly, `|S|²` is **power** (instantaneous, per frame), not energy — energy requires integration over time — but the two terms are often used loosely.

Each of the `n_mels` filters is a triangle in the frequency domain: it ramps up from zero, peaks at its center frequency, ramps back down, and overlaps its neighbors so no FFT bin is left unweighted. Unlike the time-domain Hann windows used by the STFT, these filters operate purely in the frequency domain — they weight FFT bins that already exist, they don't touch the audio signal.

The dot product of one filter with one frame's power spectrum yields one mel bin: the total power in that perceptual band at that moment. Across all `n_mels` filters and all time frames, this produces the final `(n_mels × time_frames)` matrix.

Because mel filters aggregate many FFT bins into each output bin, some spectral detail is lost — this is why mel reconstruction sounds more degraded than STFT reconstruction.

> The frequency axis scale control is hidden when Mel is selected; the mel axis is always mel-scaled.

---

## Parameters

### `Frequency Axis Scale` — *(STFT type only)*
Determines how the y-axis is rendered:

- **Linear** — uniform Hz spacing from `0` to `sample_rate / 2` (Nyquist). Harmonic overtones appear as evenly-spaced horizontal bands.
- **Log** — logarithmic spacing so each octave (doubling of frequency) occupies equal vertical height. Matches human pitch perception; melodic intervals are easier to identify.

### `n_fft` — FFT window size

Controls the fundamental **time–frequency resolution tradeoff**:

| Larger `n_fft` | Smaller `n_fft` |
|---|---|
| Finer frequency bins (`Δf = sr / n_fft`) | Coarser frequency bins |
| Wider time window, blurs fast transients | Narrower window, captures sharp attacks |

Typical values: 512 (drums, transients) → 2048 (general) → 4096 (low-frequency detail).

### `hop_length` — frame step

The number of samples the window advances between frames. Sets **time resolution**:

```
Δt = hop_length / sample_rate   (e.g. 512 / 44100 ≈ 11.6 ms)
```

Overlap between successive frames is `1 − hop_length / n_fft`. A common default is `n_fft / 4`.

### `n_mels` — mel filter banks *(Mel type only)*

The number of triangular filters in the mel filterbank. More filters preserve finer perceptual frequency resolution at the cost of a larger feature matrix. The default of 128 is standard for music; speech models often use 40–80.

For reconstruction, `n_mels` must be large enough relative to `n_fft` to keep the mel→STFT inversion stable (minimum = `ceil((n_fft // 2 + 1) / 11)`).

---

## Reconstruction

Each STFT bin `S[k, t]` is a complex number `a + bi`. From it you can derive two quantities: **magnitude** `√(a² + b²)` and **phase** `atan2(b, a)` (where in its oscillation cycle that component sits). The spectrogram only stores and displays magnitudes — phase is thrown away. To convert a spectrogram back to audio (**inversion**), you need both. The iSTFT (inverse STFT) requires a complex-valued matrix, both magnitudes and phases.

**Griffin-Lim** estimates a plausible phase by iterating between the time and frequency domains:

1. Start with a random phase estimate.
2. **Apply the magnitude constraint.** Combine the known magnitudes with the current phase estimate to form a complex matrix.
3. **Invert to the time domain (iSTFT).** For each frame, run an inverse FFT to produce a windowed time slice, then overlap-add successive slices to reconstruct a continuous waveform. Because the current phase estimate isn't yet self-consistent across frames, adjacent slices disagree where they overlap and the overlap-add averages out the disagreement — that residual error is what the next step measures.
4. Re-compute the STFT to get a new phase estimate.
5. Repeat from step 2.

The **Griffin-Lim iterations** control sets how many passes to run. More iterations reduce phase inconsistency and improve perceptual quality, with diminishing returns past ~32.
