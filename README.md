# Audio File <-> Spectrogram Conversion

A spectrogram is a time × frequency heatmap showing how the content of a signal evolves moment to moment. This project is a tool for visualizing audio files as spectrograms, and reconstructing them via Griffin-Lim.

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

---

## Example

Demos of round-trip on 60 second music clip  ([717x-chillwave.mp3](https://github.com/user-attachments/files/28295068/717x-chillwave.mp3)):

**STFT reconstruction** 

https://github.com/user-attachments/assets/9dced0ed-0e54-4cb2-b447-439e72758ea0

`n_fft=2048`, `hop_length=512`, 32 Griffin-Lim iterations: loses some quality but still gets all the ideas through.

**Mel reconstruction** 

https://github.com/user-attachments/assets/e0628275-527e-4183-83c6-6f479ad7146a

`n_fft=512`, `hop_length=128`, `n_mels=32`, 64 Griffin-Lim iterations: very muffled and degraded but recognizable.

---

## Spectrogram types

Both spectrogram types here display audio values on a decibel (dB) scale, which is fundamentally a power ratio: `dB = 10 · log₁₀(P / P_ref)`. Because `P ∝ A²`, for amplitude inputs the equivalent is `20 · log₁₀(A / A_ref)`

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

Because mel filters aggregate many FFT bins into each output bin, some spectral detail is lost and the reconstruction sounds more degraded than STFT reconstruction.


---

## Parameters

### `Frequency Axis Scale` *(display only)*
Determines how the y-axis is rendered:

- **Linear**: uniform Hz spacing from `0` to `sample_rate / 2` (Nyquist). Harmonic overtones appear as evenly-spaced horizontal bands.
- **Log**: logarithmic spacing so each octave (doubling of frequency) occupies equal vertical height. Matches human pitch perception; melodic intervals are easier to identify.

> <sub> Hidden when Mel is selected; the mel axis is always mel-scaled </sub>

### `FFT Window Size (n_fft)` *(display + reconstruction)*

The number of samples analyzed by a single FFT. Controls the fundamental **time–frequency resolution tradeoff**:

- **Frequency resolution:** `Δf = sample_rate / n_fft` Hz per bin
- **Window duration:** `n_fft / sample_rate` seconds — the time interval each frame summarizes

| Larger `n_fft` | Smaller `n_fft` |
|---|---|
| Finer frequency bins | Coarser frequency bins |
| Wider time window, blurs fast transients | Narrower window, captures sharp attacks |

Typical values: 512 (drums, transients) → 2048 (general) → 4096 (low-frequency detail). Powers of two are conventional because the FFT is fastest there.

### `Hop Length (hop_length)` *(display + reconstruction)*

The number of samples the window advances between frames. Sets time resolution along the spectrogram x-axis:

```
Δt = hop_length / sample_rate   (e.g. 512 / 44100 ≈ 11.6 ms)
```

The default `hop_length = n_fft / 4` (75% overlap) satisfies the COLA condition for the Hann window, so iSTFT and Griffin-Lim can reconstruct cleanly. Use ≤ 50% overlap only if you don't need resynthesis.

### `Mel Bins (n_mels)` *(display + reconstruction)*

The number of triangular filters in the mel filterbank. More filters preserve finer perceptual frequency resolution at the cost of a larger feature matrix. The default of 128 is standard for music; speech models often use 40–80.

For reconstruction, `n_mels` must be large enough relative to `n_fft` to keep the mel→STFT inversion stable (minimum = `ceil((n_fft // 2 + 1) / 11)`).

> Hidden when STFT is selected

### `dB Reference` *(display only)*

Sets which value maps to 0 dB on the color scale.

- **Per-clip** (default): the loudest bin in the clip is the reference. Max is always 0 dB; everything else is negative. Each clip uses the full color range, but values are not comparable across clips.
- **Full scale (dBFS)**: 0 dB = digital full-scale amplitude (1.0). Values are comparable across clips, but quiet recordings render dim because their peak sits well below 0.

### `Griffin-Lim Iterations` *(reconstruction only)*

Number of passes to run through the iterative reconstruction process, see details below. More iterations reduce phase inconsistency and improve perceptual quality, with diminishing returns.

---

## Reconstruction

Each STFT bin `S[k, t]` is a complex number `a + bi`. From it you can derive two quantities: **magnitude** `√(a² + b²)` and **phase** `atan2(b, a)` (where in its oscillation cycle that component sits). The spectrogram only displays magnitudes, but we need both for the iSTFT (inverse STFT) "inversion" process.

**Griffin-Lim** estimates a plausible phase by iterating between the time and frequency domains:

1. Start with a random phase estimate.
2. **Apply the magnitude constraint.** Combine the known magnitudes with the current phase estimate to form a complex matrix.
3. **Invert to the time domain (iSTFT).** For each frame, run an inverse FFT to produce a windowed time slice, then overlap-add successive slices to reconstruct a continuous waveform. Because the current phase estimate isn't yet self-consistent across frames, adjacent slices disagree where they overlap and the overlap-add averages out the disagreement — that residual error is what the next step measures.
4. Re-compute the STFT to get a new phase estimate.
5. Repeat from step 2.
