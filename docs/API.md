# Library API: `representation_intelligence`

Reference for the audio library. Overview in the [README](../README.md), experiments in [EXPERIMENTS.md](EXPERIMENTS.md).

The current Python library is the first practical application of the idea, for audio only. It reads physical descriptors of each audio frame, places the frame on a learned 2D state map, predicts which representations are blind there, and recommends DSP parameters.

It also runs STFT, ACF and Cepstrum side by side on every frame and reports where they disagree (`state.disagreement`, on by default; see [Cross-representation disagreement](#cross-representation-disagreement)).

> **Known gap:** the library compares representations but does not yet learn from their disagreements. The pattern model exists only as an interface in [`src/core`](../src/core/interfaces.py) and in the Exp 043 script, and the recommended parameters still come from the state map alone, not from the disagreement. See [VISION.md](VISION.md#6-roadmap-direction).

## Install

```bash
pip install git+https://github.com/SpacekidLabs/representation-fragility-lab.git
```

## Basic usage

```python
from representation_intelligence import RepresentationIntelligenceEngine

engine = RepresentationIntelligenceEngine()
state = engine.analyze(audio_frame, sr=16000)

print(f"Region: {state.region}") # e.g. 'noise_collapse', 'periodic_harmonic'
print(f"Recommended Window: {state.recommended_window} samples")
```

## Engine output

The `engine.analyze(frame, sr)` call returns a `FrameworkState` object containing:

| Property | Type | Description |
| :--- | :--- | :--- |
| `state.coordinate` | `tuple[float, float]` | $(z_1, z_2)$ coordinates mapping Order ↔ Disorder and Harmonic ↔ Transient. |
| `state.region` | `str` | Semantic region: `periodic_harmonic`, `noise_collapse`, `transient_overloaded`, `smooth_lowpass`, or `transition_zone`. |
| `state.safe_representations` | `list[str]` | List of valid representation algorithms (from `stft`, `acf`, `cepstrum`, `cqt`, `wavelet`). |
| `state.recommended_window` | `int` | Optimal window size (1024, 2048, or 4096 samples). |
| `state.recommended_parameters`| `dict` | Pre-calibrated parameters for `denoising`, `pitch_tracking`, and `onset_detection`. |
| `state.disagreement` | `DisagreementReport` | Cross-representation disagreement: STFT, ACF and Cepstrum run side by side on the frame (see below). `None` when `analyze(..., compute_disagreement=False)`. |


## Cross-representation disagreement

The core idea of the project: run several lenses on the same frame and treat their disagreement as information. Each representation is reduced to a shared form (`f0`, `confidence`, `voiced`) and every pair is compared.

```python
from representation_intelligence import analyze_disagreement

report = analyze_disagreement(frame, sr=16000)      # or engine.analyze(...).disagreement

report.readings["cepstrum"]          # RepresentationReading(name, f0, confidence) + .voiced
report.pair("acf", "cepstrum")       # cents, octave_related, pitch_term, confidence_gap, voicing_conflict, score
report.overall                       # mean pair score, 0 (all agree) .. 1
report.divergence                    # per representation: how much it disagrees with the others
report.outlier                       # the lens most likely in a blind spot, or None
report.consensus_f0                  # confidence-weighted median of the agreeing voiced readings (0.0 without a quorum of 2)
```

Pair score = `0.5 * min(cents / 1200, 1) + 0.5 * |confidence_a - confidence_b|`. Definitions and calibration notes are in [`src/framework/disagreement.py`](../src/framework/disagreement.py); tests in [`tests/test_disagreement.py`](../tests/test_disagreement.py) (`pytest tests`).

Under the hood this runs on the medium-agnostic core ([`src/core`](../src/core), [design note](core_interfaces.md)), with audio as the first medium ([`src/framework/audio.py`](../src/framework/audio.py)). Five pitch lenses are available: STFT, ACF and Cepstrum (the default) plus CQT and wavelet:

```python
from representation_intelligence import make_audio_analyzer, analyze_disagreement, ALL_LENSES

analysis = make_audio_analyzer(ALL_LENSES).analyze(frame, {"sr": 22050})
analysis.report.consensus, analysis.report.outlier
analyze_disagreement(frame, sr=22050, representations=ALL_LENSES)   # same, in the original report format
```

Adding lenses is not free: STFT, CQT and wavelet share the missing-fundamental blind spot and can outvote the other two (see the design note), so the default stays at three.

## Examples

### Adaptive YIN pitch tracking
Improve YIN tracking under extreme noise blocks and transients by dynamically scaling the window size and trough threshold based on engine state.

```python
import librosa
from representation_intelligence import RepresentationIntelligenceEngine

engine = RepresentationIntelligenceEngine()
st = engine.analyze(frame, sr=22050)

# Dynamically adapt YIN parameters
pitch = librosa.yin(
    frame, fmin=80, fmax=500, sr=22050,
    frame_length=st.recommended_window,
    hop_length=st.recommended_window,
    trough_threshold=st.recommended_parameters["pitch_tracking"]["yin_trough"],
    center=False
)[0]
```
* **Performance Gain**: **+5.8% tracking accuracy improvement** under noise/transient blocks, with **zero regressions** on clean audio.

### Adaptive Wiener denoising
Adapt local window filter size and noise power estimations in a standard Wiener filter on-the-fly to prevent transient smearing.

```python
import scipy.signal
from representation_intelligence import RepresentationIntelligenceEngine

engine = RepresentationIntelligenceEngine()
st = engine.analyze(frame, sr=16000)

# Adapt parameters based on physical region
if st.region == "noise_collapse":
    mysize, noise_power = 17, np.var(frame) * 0.75
elif st.region == "transient_overloaded":
    mysize, noise_power = 3, 1e-4
else:
    mysize, noise_power = 7, None

denoised_frame = scipy.signal.wiener(frame, mysize=mysize, noise=noise_power)
```
* **Performance Gain**: **+2.6% denoising performance improvement** ($\Delta P = +0.0259$) with no phase artifacts.
