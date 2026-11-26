# Vision — Multi-Representation Blind-Spot Intelligence

> **Every representation of a signal is a lens with blind spots.
> Look through many lenses at once, compare what they see, and the
> disagreements themselves reveal what the signal really is.**

This document is the north star of the project. Everything in this repository —
experiments, the framework, the plugins — should be read as a step toward it.

---

## 1. The Goal

Build a **medium-agnostic, multi-representation system** that:

1. **Represents** any signal — audio, image, video, text, sensor data — through
   several complementary representations at the same time.
2. **Detects blind spots** — knows, for each representation, where and when it
   is unreliable on *this* signal.
3. **Compares** the representations against each other. Agreement is evidence.
   Disagreement is information: it locates a blind spot and tells us what kind
   it is.
4. **Finds patterns** in those agreements and disagreements, learning a
   structured map of how and why representations fail.
5. **Understands the signal better** than any single representation can, by
   combining the lenses according to what each one can and cannot see.

Audio is the **first medium**, not the destination.

---

## 2. Core Principles

| Principle | Meaning |
|---|---|
| **No representation is complete** | Every transform makes assumptions (stationarity, periodicity, locality, smoothness…). Where an assumption breaks, the representation is blind. |
| **Failures are structured, not random** | Each representation has a specific, repeatable failure signature. Blind spots can be mapped, predicted and deliberately probed. |
| **Disagreement is signal** | When representations disagree, that difference carries information about the signal that none of them carries alone. |
| **Patterns in failure are knowledge** | The geometry of where representations fail (and fail together) is itself a description of the signal's state. |
| **Self-awareness over guessing** | A system that knows when it is blind (and abstains, holds, or reroutes) beats one that always answers. |
| **Medium-agnostic core** | The comparison and pattern-finding machinery must not depend on audio. Each medium plugs in its own representations. |

---

## 3. The Loop

```
                 ┌──────────────────────────────────────────┐
  Signal  ──►    │  Representation set (per medium)         │
 (any medium)    │  R1   R2   R3   …   Rn                    │
                 └──────────────┬───────────────────────────┘
                                ▼
                 ┌──────────────────────────────────────────┐
                 │  Blind-spot detection                    │
                 │  per-representation confidence / safety  │
                 └──────────────┬───────────────────────────┘
                                ▼
                 ┌──────────────────────────────────────────┐
                 │  Cross-representation comparison         │
                 │  agreement / disagreement maps           │
                 └──────────────┬───────────────────────────┘
                                ▼
                 ┌──────────────────────────────────────────┐
                 │  Pattern discovery                       │
                 │  structure in failures & disagreements   │
                 └──────────────┬───────────────────────────┘
                                ▼
                 ┌──────────────────────────────────────────┐
                 │  Understanding & action                  │
                 │  fused estimate · signal state ·         │
                 │  routing · abstain when blind            │
                 └──────────────────────────────────────────┘
```

---

## 4. Vocabulary

- **Medium** — a kind of signal: audio, image, video, text, time-series…
- **Representation** — a transform of the signal that exposes some structure
  (e.g. STFT, autocorrelation, cepstrum, wavelets for audio; pixels, DCT/FFT,
  wavelets, edges, learned embeddings for images).
- **Blind spot** — a region of signal conditions where a representation's
  assumptions break and its output becomes unreliable.
- **Disagreement** — a measurable difference between what two or more
  representations say about the same signal.
- **Failure / state map** — the learned, structured space in which signals,
  blind spots and disagreements are organised.

---

## 5. Where We Are — Audio as Medium #1

The 41 audio experiments are the proof of concept for this vision:

| Vision component | Status in audio | Evidence |
|---|---|---|
| Multiple representations | ✅ STFT, ACF, Cepstrum, CQT, Wavelet, Mel | Exp 001–013, 028, 031 |
| Blind-spot mapping | ✅ Each representation's failure signature mapped; probe signals that blind one lens and spare others | Exp 001–013 |
| Self-confidence / failure forecasting | ✅ Per-representation confidence predicts failure before it happens | Exp 015–016 |
| Comparison & fusion | ✅ Explicit disagreement (`src/framework/disagreement.py`): it predicts blind spots (AUC 0.95–0.99) and the consensus beats any single lens | Exp 014, 017–018, 042 |
| Pattern discovery | ✅ Disagreement patterns: which lenses fail together and when | Exp 025–031, 043 |
| Using it in practice | ✅ Confidence-gated tuner, adaptive plugins, external algorithms | Exp 019–024, 033–041 |
| Medium-agnostic core | ✅ Interfaces and generic loop in `src/core` ([design note](core_interfaces.md)); audio runs on it with 5 lenses (STFT, ACF, Cepstrum, CQT, wavelet) | Issues #4, #5 |
| Second medium: images | 🟡 `src/image_medium`: 5 orientation lenses on the same core. Disagreement predicts blind spots (AUC 0.96–0.98), but correlated failures keep the consensus from beating the best single lens. Real photos (BSDS500): the warning holds (AUC 0.87–0.92), fusion still does not beat the best lens | Exp 044, 045, issue #6 |
| Real-world validation | 🟡 Key claims re-run on vocadito, MDB-stem-synth and BSDS500. The disagreement warning holds in both media; the consensus gain holds on one of two audio datasets; the framework's pitch-tracker gain does not hold; unpitched real frames fool single lenses; with the quorum (Exp 046) the consensus no longer follows them | Exp 045, 046, issues #7, #20 |

**Honest gaps:** disagreement reliably *warns* that a lens is blind, in both media
and on real data (Exp 045). Turning that warning into a better *answer* is the
open problem: it works on synthetic audio and on MDB-stem-synth, ties on real
singing, and fails on images, where lenses fail in families. The consensus also
no longer trusts a single confident lens: a quorum of two agreeing lenses
(Exp 046) cuts false answers on unpitched / unoriented input to 0–7%, and its
answers are more accurate, at the cost of abstaining more under heavy noise.
Fusion that knows which lenses share blind spots is next.

---

## 6. Roadmap Direction

High-level steps; concrete tickets will be opened against these.

1. **Make disagreement first-class (audio).** Run representations in parallel,
   compute explicit agreement/disagreement measures, and show that
   disagreement predicts blind spots and errors.
2. **Pattern discovery on disagreement.** Learn the structure of
   disagreements (which lenses fail together, under which conditions) instead
   of only raw signal descriptors.
3. **Medium-agnostic core.** Define abstract interfaces — `Medium`,
   `Representation`, `BlindSpotDetector`, `Comparator`, `PatternModel` — with
   audio as the first implementation.
4. **Second medium.** Implement images (or another medium) on the same core
   to prove the idea generalises beyond audio.
5. **Real-world validation.** Move from mostly synthetic signals to real
   datasets with ground truth, in every medium.
6. **Applications.** Keep shipping practical tools (e.g. the adaptive tuner)
   as demonstrations of the core, not as the core itself.

---

## 7. Non-Goals

- Not a single "best representation" — the point is that no single one exists.
- Not an audio-only parameter recommender — that is one application.
- Not a black-box model that hides *why* it trusts a representation —
  blind spots and disagreements must stay inspectable.
