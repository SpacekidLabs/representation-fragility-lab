# Exp042: Mel-spectrogram and Chroma Blind-Spot Atlas

## Goal
Extend the representation-fragility methodology to Mel-spectrogram and Chroma
representations under additive noise. Rank fragility across the standard signal
atlas (sine, harmonic stack, FM tone, chirp, percussive, noise).

## Signal Classes Mapped
- `sine`
- `harmonic_stack`
- `fm_tone`
- `chirp`
- `percussive`
- `noise`

## SNR Sweep Range
`{inf, 20, 10, 5, 0, -5}` dB

## Key Findings

### 1. Chroma is fragile for tonal signals under noise
For pitched signals, Chroma similarity drops from ~1.0 to ~0.88–0.89 at -5 dB.
This confirms that chroma is **harmony/periodicity-sensitive** and collapses
as SNR degrades.

### 2. Chroma is stable for noise-like and non-tonal signals
- `noise`: Chroma ~0.993 at -5 dB
- `percussive`: Chroma ~0.994 at -5 dB

Interpretation: when there is no stable pitch content, Chroma has little to
corrupt, so it appears “robust”.

### 3. Mel is generally more stable under noise than Cepstrum
Mel-spectrogram degrades gracefully:
- Harmonic stack: 0.986 → 0.963 (−5 dB)
- Sine: 0.993 → 0.982 (−5 dB)
- Percussive: 0.982 → 0.768 (−5 dB)

Percussive is the weak point for Mel.

### 4. Chirp and FM Tone show similar mid-range behavior
Chroma remains near 1.0 down to 5 dB, then drops at 0 / -5 dB.
Mel degrades gently from ~0.993 to ~0.980.

## Comparison With Other Representations

| Representation | noise | harmonic_stack | percussive | chirp |
|-----------------|------:|---------------:|-----------:|------:|
| Cepstrum        | +5?  | high first, then collapse | low from start | medium first, then low |
| Mel             | high  | high           | **lowest** | high  |
| Chroma          | high  | fails at -5 dB | high       | fails at -5 dB |

## Why This Matters
- **Chroma fractures** when harmonic structure degrades;
- **Mel degrades lower when energy loses spectral focus** (percussive);
- This gives a concrete taxonomy for when each representation fails.

## Artifacts
- `results/exp042_mel_chroma_blind_spots/*.png`
- `results/exp042_mel_chroma_blind_spots/*.json`

## Reproducing
```bash
source /Users/user/venvs/refrag_lab/bin/activate
python src/experiments/exp042_mel_chroma_blind_spots.py --signal harmonic_stack --save
```

## Next Step
- Run the same sweep for `pitch_shift`, `time_stretch`, and `filter` sweeps.
- Add automatic summary figure: Mel vs Chroma fragility by signal.
