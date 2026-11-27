# Representation Fragility Lab

> **Every representation of a signal is a lens with blind spots.**
> Look through many lenses at once, compare what they see, and the
> disagreements reveal what the signal really is.

Audio is the first medium, images the second, periodic sensor signals the third. Full goal and roadmap: [docs/VISION.md](docs/VISION.md).

## The loop

```mermaid
flowchart LR
    S[Signal] --> M[Medium<br/>validate, skip empty]
    M --> R1[Lens 1]
    M --> R2[Lens 2]
    M --> R3[Lens n]
    R1 & R2 & R3 --> D[Blind-spot detector<br/>per reading]
    D --> C[Comparator<br/>distance, outlier, consensus]
    C --> P[Pattern model<br/>which lenses fail together]
    P --> A[Answer, or abstain<br/>when fewer than two<br/>trusted lenses agree]
```

One medium-agnostic core ([`src/core`](src/core), [design note](docs/core_interfaces.md)). A medium only adds its lenses and a distance function.

| Medium | Quantity | Lenses |
|---|---|---|
| Audio | pitch (f0) | STFT, ACF, Cepstrum (+ CQT, wavelet); CREPE as an opt-in learned lens |
| Images | dominant orientation | structure tensor, HOG, FFT, Gabor, Radon; a CNN read-out as an opt-in learned lens |
| Sensor time series | rate of a periodic signal (e.g. heart rate) | FFT, ACF, wavelet, peak counter |

## Headline results

### 1. Disagreement predicts when a lens is wrong

ROC AUC for predicting a gross error (0.5 = chance):

| | Own confidence | Disagreement |
|---|---|---|
| Audio, 3 lenses (Exp 042) | 0.81 – 0.88 | **0.95 – 0.99** |
| Images, 5 lenses (Exp 044) | 0.59 – 0.89 | **0.96 – 0.98** |

![Exp 042](results/exp042_disagreement_prediction.png)

### 2. Comparing lenses gives better answers (audio)

```mermaid
xychart-beta
    title "Gross error rate when answering (%), Exp 042"
    x-axis ["Cepstrum", "Routed lens", "ACF", "STFT", "Consensus"]
    y-axis "%" 0 --> 30
    bar [26.7, 9.1, 6.9, 4.3, 1.0]
```

The consensus abstains on 5.6% of frames, almost all at −5 dB SNR where no lens is confident.

### 3. Lenses fail in families

| Pair (audio, Exp 043) | Fail together vs chance |
|---|---|
| ACF + Cepstrum | **2.1×**, both measure periodicity |
| STFT + ACF | 1.6× |
| STFT + Cepstrum | **0.08×**, the most complementary pair |

Disagreement patterns explain **2.5×** more of the true failure structure than the input-only state map (NMI 0.546 vs 0.223).

![Exp 043](results/exp043_disagreement_patterns.png)

On images the same split appears between local (gradient) and global (spectral, projection) lenses, and correlated failures outvote the right lens: the consensus (9.8%) loses to the best single lens (Radon, 4.8%).

![Exp 044](results/exp044_image_disagreement.png)

### 4. On real data (Exp 045)

The key claims re-run on vocadito (real singing), MDB-stem-synth (230 resynthesised performances) and BSDS500 (photos with human boundaries):

| Claim | Real data |
|---|---|
| Disagreement predicts when a lens is wrong | **Holds**: AUC 0.86–0.99 (audio), 0.87–0.92 (photos) |
| The consensus is a better answer (audio) | Holds on MDB-stem-synth (7.3% vs 9.9%); a tie on vocadito |
| ACF + Cepstrum share a blind spot | Holds (2.6–3.0×); STFT + Cepstrum are no longer complementary (0.7–0.85×) |
| The framework's pitch tracker beats fixed YIN | **Does not hold**: a loss. Exp 054 traces it to the tracker's median filter, which lags vibrato and note changes, and to window choices that hurt under noise. The claim is retired. A quorum of the plain lenses with a YIN fallback (`QuorumPitchTracker`) does beat fixed YIN on both real datasets |
| No confident answer when there is no pitch | **Failed**: breaths and consonants fool single lenses (36–99% of unvoiced frames). Fixed by a quorum of two agreeing lenses (Exp 046): 0–7%, at the cost of more abstentions in heavy noise |

Full table with every dataset, metric and confidence interval: [Exp 045 in docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#exp-045--do-the-claims-survive-real-recordings-and-photos).

![Exp 045](results/exp045_real_data.png)

## How we got here

```mermaid
flowchart TB
    subgraph R1["Audio: lenses and their blind spots"]
        direction LR
        A["Exp 001–013<br/>Blind-spot atlas"] --> B["Exp 014–031<br/>Failure manifold,<br/>state space"]
        B --> C["Exp 032–041<br/>Framework and<br/>DSP gains"]
        C --> D["Exp 042–043<br/>Disagreement as<br/>information"]
    end
    subgraph R2["A second medium, real data, a quorum"]
        direction LR
        E["Exp 044<br/>Second medium:<br/>images"] --> G["Exp 045<br/>Real data"]
        G --> H["Exp 046<br/>Consensus<br/>quorum"]
        H --> I["Exp 047–051<br/>Fusion, chooser,<br/>abstention, probing,<br/>time"]
    end
    subgraph R3["Learned lenses, a third medium, what each lens is worth"]
        direction LR
        J["Exp 052<br/>Learned lenses"] --> K["Exp 053<br/>Third medium:<br/>sensor signals"]
        K --> L["Exp 054–056<br/>Tracker retired,<br/>lens value,<br/>lead lens"]
        L --> F["Next: probing #30,<br/>photo lens #32,<br/>Lean proofs #11"]
    end
    R1 --> R2 --> R3
```

| Exp | Question | Answer |
|---|---|---|
| 047 | Does knowing which lenses fail together improve fusion? | On images, by 2.6–3.1 pts; it still cannot beat the best single lens |
| 048 | Can a model pick the right lens per frame? | Yes on real singing (1.8% vs 4.2% for the best lens), but it cannot tell when there is no pitch |
| 049 | Can a rule answer more than the quorum without more false answers? | A learned gate can, on MDB-stem-synth and photos; on real singing nothing beats the quorum |
| 050 | Can perturbing the signal settle a disagreement? | Low-pass probing cuts gross error on disagreement frames to 3.9% / 5.6% (from 14.5% / 17.6%), untrained |
| 051 | Does the recent history of disagreement warn earlier? | A little, 4–8 frames ahead, mostly for the STFT |
| 052 | Is a learned lens a new family? | CREPE fails on its own and is far more accurate; it should answer while the classic lenses warn |
| 053 | Does the core carry a third medium? | Unchanged; the warning holds (AUC 0.90–0.93), but three lenses share a blind spot and outvote the peak counter |
| 054 | Why does the pitch tracker lose on real data? | Its median filter lags vibrato and note changes; the claim is retired and a quorum tracker beats fixed YIN |
| 055 | Which lenses are worth keeping? | A lens's value follows its coverage (right where the others cannot agree), not its independence; HOG and Cepstrum add almost nothing |
| 056 | Should the best lens answer alone, with the others as a warning? | Yes where one lens is clearly better (−2 to −28 pts vs the quorum; opt-in `lead=` in the core); the others warn well in audio, weakly on ECG |

| | | | |
|---|---|---|---|
| ![](results/exp004_compare_all.png) | ![](results/exp012_targeted_probing.png) | ![](results/exp027_failure_manifold.png) | ![](results/exp031_assumption_surfaces.png) |
| Exp 004: three lenses, three failure signatures | Exp 012: probes that blind one lens | Exp 027: failures form a 2-D manifold | Exp 031: nested safe zones per lens |
| ![](results/exp015_self_aware_tracking.png) | ![](results/exp023_retune_atlas.png) | ![](results/exp034_framework_validation.png) | ![](results/exp041_external_validation.png) |
| Exp 015: confidence warns before failure | Exp 023: retune dynamics | Exp 034: one engine, five DSP tasks | Exp 041: gains on external algorithms |

Every experiment, with numbers and caveats: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pip install -r requirements-learned.txt   # optional: the learned lenses (CREPE, CNN), needs torch
python -m pytest tests
```

```python
from representation_intelligence import analyze_disagreement

report = analyze_disagreement(frame, sr=22050)   # frame: 1-D float array
report.consensus_f0                  # fused pitch, Hz
report.outlier                       # the lens most likely in a blind spot, or None
report.pair("acf", "cepstrum").cents # how far two lenses disagree
```

Library reference and examples: [docs/API.md](docs/API.md). Running each experiment: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#running-the-experiments).

## Repository

```
src/core/            medium-agnostic interfaces and loop
src/framework/       audio: lenses, disagreement, state-space engine, plugins, learned lens (CREPE)
src/image_medium/    images: five orientation lenses, learned lens (CNN read-out)
src/timeseries_medium/  sensor time series: four rate lenses
src/representations/, signals/, perturbations/, metrics/   building blocks of the early experiments (Exp 001–013)
src/vst_prototype/   C++ plugin prototype
src/experiments/     one script per experiment
data/                dataset setup (data/README.md); downloads are not committed
results/             plots and summaries
docs/                vision, experiments, API, design notes
tests/               pytest
lean/                Lean 4 proofs of core statements (issue #11)
```

## Known limits

- Most experiments use synthetic signals. Exp 045 re-tests the key claims on real data: the warning signal holds and the framework's pitch-tracker gain does not (Exp 054 says why; the quorum tracker that replaces it beats fixed YIN). Unpitched real frames fooled the old consensus; the quorum (Exp 046) fixes that at the cost of more abstentions.
- More lenses are not more independent votes: lenses that share a blind spot outvote the others (images, and ECG heart rate in Exp 053). Turning disagreement into a better answer works on synthetic audio and MDB-stem-synth, ties on real singing, and fails on images and ECG.
- Not every lens earns its place (Exp 055): HOG and Cepstrum add almost nothing to the answer, and a lens's value follows how often it is right where the others cannot agree, not how independent its blind spots are.
- The framework's DSP gains are small and depend on the baseline (see Exp 035 and Exp 041 in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)).
