# Audio Spectrogram Generator

A Gradio demo for visualizing audio as spectrograms and reconstructing audio from them via Griffin-Lim.

```bash
pip install -r requirements.txt
python -m src.app
# → http://localhost:7860
```

---

## Spectrogram types

All types express amplitude on a **decibel (dB) scale**:

```
dB = 20 · log₁₀(|S| / max|S|)
```

where `|S|` is the magnitude of each time-frequency bin, normalized so the loudest bin is always 0 dB.

### STFT

The Short-Time Fourier Transform divides the signal into short, overlapping frames and computes the FFT on each. Before the FFT, each frame is multiplied by a **Hann window** — a smooth bell-shaped curve that tapers to zero at both ends. Without windowing, the hard cut-off at each frame boundary introduces artificial discontinuities that leak energy across frequency bins (spectral leakage). The Hann window eliminates those edges, confining each sinusoidal component to its true frequency and a narrow set of neighboring bins.

```
S[k, t] = Σₙ x[n] · w[n − t·H] · e^(−j2πkn/N)
```

where `N` = `n_fft`, `H` = `hop_length`, and `w` is the Hann window. The magnitude `|S[k, t]|` is plotted.

The **Frequency Axis Scale** control determines how the y-axis is rendered:

- **Linear** — uniform Hz spacing from 0 to `sample_rate / 2` (Nyquist). Each bin covers exactly `sample_rate / n_fft` Hz. Harmonic overtones appear as evenly-spaced horizontal bands.
- **Log** — logarithmic spacing so each octave (doubling of frequency) occupies equal vertical height. Matches human pitch perception; melodic intervals are easier to identify.

### Mel

Applies a bank of `n_mels` triangular filters to the **power spectrogram** — the squared magnitude of the STFT (`|S[k, t]|²`), which measures energy rather than amplitude. Squaring emphasizes louder components and matches how acoustic energy is physically defined.

The Hann window and the mel filterbank are both weighting operations, but they act in orthogonal dimensions. The Hann window operates in the **time domain**: it weights the raw audio samples within each frame before the FFT, shaping what a single snapshot of the signal looks like. The mel filters operate in the **frequency domain**: they weight the FFT bins that the STFT has already produced, running across the frequency axis for each time frame independently. Each triangular filter covers a contiguous range of FFT bins, ramping up from zero, peaking, then ramping back down, and adjacent filters overlap so no frequency region falls through the cracks. The dot product of a filter with a frame's power spectrum yields one mel bin — a single number representing the total energy in that perceptual frequency band at that moment in time. Applied across all `n_mels` filters and all time frames, this produces the final `(n_mels × time_frames)` matrix.

The filters are spaced on the **mel scale**:

```
mel(f) = 2595 · log₁₀(1 + f / 700)
```

The mel scale approximates how the cochlea resolves frequencies: dense filter spacing at low frequencies and coarse spacing at high frequencies. The result is a compact, perceptually-motivated representation widely used as input features for speech and music models.

Because mel filters aggregate many FFT bins into each output bin, some spectral detail is lost — this is why mel reconstruction sounds more degraded than STFT reconstruction.

> The frequency axis scale control is hidden when Mel is selected; the mel axis is always mel-scaled.

---

## Parameters

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

For reconstruction, `n_mels` must be large enough relative to `n_fft` to keep the mel→STFT inversion stable (minimum ≈ `(n_fft // 2 + 1) / 11`).

### Griffin-Lim iterations *(reconstruction only)*

Each STFT bin `S[k, t]` is a complex number `a + bi`. From it you can derive two quantities: **magnitude** `√(a² + b²)` (how loud that frequency is) and **phase** `arctan(b / a)` (where in its oscillation cycle that frequency component sits at that moment). The spectrogram only stores and displays magnitudes — phase is thrown away. The spectrogram only stores and displays magnitudes — phase is thrown away. To convert a spectrogram back to audio (**inversion**), you need both: the iSTFT (inverse STFT) requires a complex-valued matrix, not just magnitudes. With phase missing, a direct inversion is impossible.

**Griffin-Lim** estimates a plausible phase by iterating between the time and frequency domains:

1. Start with a random phase estimate.
2. Apply the magnitude constraint: replace magnitudes with the known spectrogram values, keep the estimated phase.
3. Invert to the time domain (iSTFT).
4. Re-compute the STFT to get a new phase estimate.
5. Repeat from step 2.

More iterations reduce phase inconsistency and improve perceptual quality, with diminishing returns past ~32.
