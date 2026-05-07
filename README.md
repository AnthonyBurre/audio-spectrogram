# Audio File <-> Spectrogram Conversion

A tool for visualizing audio as spectrograms and reconstructing audio from them via Griffin-Lim.

```bash
pip install -r requirements.txt
python -m src.app
# → http://localhost:7860
```

---

## Spectrogram types

All types display values on a **decibel (dB) scale**, normalized so the loudest bin in the clip is always 0 dB. dB values here are *relative to that maximum* - they are not absolute SPL.

For STFT (amplitude form):

```
dB = 20 · log₁₀(|S| / max|S|)
```

For Mel (power form, since the mel filterbank is applied to `|S|²`):

```
dB = 10 · log₁₀(P / max P)
```

The two forms are numerically equivalent when `P = |S|²`, since `20·log₁₀|S| = 10·log₁₀|S|²`. Power-dB is the more fundamental definition; amplitude-dB exists as a shorthand for it.
Strictly, `|S|²` is **power** (instantaneous, per frame), not energy — energy requires integration over time — but the two terms are often used loosely.

### STFT

The Short-Time Fourier Transform divides the signal into short, overlapping frames and computes the FFT on each. Before the FFT, each frame is multiplied by a Hann window — a bell-shaped curve that tapers to zero at both ends. Without windowing, the hard cut-off at each frame boundary introduces artificial discontinuities that leak energy across frequency bins (spectral leakage). The Hann window smooths those edges so energy from a single sinusoid still spreads across a few bins, but no longer leaks far across the spectrum.

The **Frequency Axis Scale** control determines how the y-axis is rendered:

- **Linear** — uniform Hz spacing from 0 to `sample_rate / 2` (Nyquist). Each bin covers exactly `sample_rate / n_fft` Hz. Harmonic overtones appear as evenly-spaced horizontal bands.
- **Log** — logarithmic spacing so each octave (doubling of frequency) occupies equal vertical height. Matches human pitch perception; melodic intervals are easier to identify.

### Mel

Applies a bank of `n_mels` triangular filters to the **power spectrogram** — the squared magnitude of the STFT (`|S[k, t]|²`). Working in power is the natural choice here: acoustic power is proportional to the square of pressure, and powers from incoherent sources add, so summing FFT-bin powers within each mel filter is physically meaningful in a way that summing amplitudes is not.

Unlike the time domain Hann windows, the mel filters operate in the **frequency domain**: they weight the FFT bins that the STFT has already produced, running across the frequency axis for each time frame independently. Each triangular filter covers a contiguous range of FFT bins, ramping up from zero, peaking, then ramping back down, and adjacent filters overlap so no frequency region falls through the cracks. The dot product of a filter with a frame's power spectrum yields one mel bin — a single number representing the total power in that perceptual frequency band at that moment in time. Applied across all `n_mels` filters and all time frames, this produces the final `(n_mels × time_frames)` matrix.

The filters are spaced on the **mel scale**, with dense spacing at low frequencies and coarse spacing at high frequencies. There is no single canonical mel formula — several variants are in common use. A frequently-cited form (HTK / O'Shaughnessy) is:

```
mel(f) = 2595 · log₁₀(1 + f / 700)
```

`librosa` uses **Slaney's** variant by default (piecewise: linear below 1 kHz, logarithmic above), which differs from the formula above mostly in the low-frequency region. To switch to the HTK form, pass `htk=True` to `librosa.feature.melspectrogram`.

The mel scale was derived from **subjective pitch-perception experiments** (Stevens, Volkmann & Newman, 1937) — it approximates *perceived pitch*, not cochlear filter widths. Auditory filter resolution on the basilar membrane is better described by the **Bark** or **ERB** scales, which are the right reference points for psychoacoustic and auditory-modeling work. Mel persists in speech and music ML largely by historical convention, but works well in practice as a compact, perceptually-motivated feature.

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

Each STFT bin `S[k, t]` is a complex number `a + bi`. From it you can derive two quantities: **magnitude** `√(a² + b²)` and **phase** `atan2(b, a)` (where in its oscillation cycle that component sits). The spectrogram only stores and displays magnitudes — phase is thrown away. To convert a spectrogram back to audio (**inversion**), you need both. The iSTFT (inverse STFT) requires a complex-valued matrix, both magnitudes and phases.

**Griffin-Lim** estimates a plausible phase by iterating between the time and frequency domains:

1. Start with a random phase estimate.
2. Apply the magnitude constraint: replace magnitudes with the known spectrogram values, keep the estimated phase.
3. Invert to the time domain (iSTFT).
4. Re-compute the STFT to get a new phase estimate.
5. Repeat from step 2.

More iterations reduce phase inconsistency and improve perceptual quality, with diminishing returns past ~32.
