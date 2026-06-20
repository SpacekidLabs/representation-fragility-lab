# Issue #9: Automated Blind Spot Discovery

## Goal
Construct an automated search loop to discover new representation blind spots—specifically finding the smallest perturbation vector that collapses a target representation (ACF) while leaving other representations (STFT and Cepstrum) intact.

---

## Results

We implemented a random search optimizer over a 4D parameter space:
- Pitch shift ($\pm 50$ cents)
- Time shift ($0$ to $50$ ms)
- Phase shift ($0$ to $2\pi$ rad)
- Noise std dev ($0$ to $0.05$)

### Discovered Adversarial Parameter Set (ACF Blind Spot)
The search loop automatically discovered the following optimal perturbation vector:
- **Pitch Shift**: $7.8653$ cents
- **Time Shift**: $13.8144$ ms
- **Phase Shift**: $1.5206$ rad
- **Noise Std Dev**: $0.0003$

### Resulting Similarities
Under this discovered perturbation:
- **STFT Similarity**: `0.9872` (Constraint: $> 0.95$ - **PASSED**)
- **Cepstrum Similarity**: `0.9637` (Constraint: $> 0.95$ - **PASSED**)
- **ACF Similarity**: `0.0378` (Objective: **MINIMIZED/DESTROYED**)

---

## Theoretical Significance

This result proves that a multidimensional combination of tiny, perceptually negligible changes can selectively collapse the Autocorrelation representation. 

The mechanism exploits **sub-bin pitch translation** combined with a **micro-time shift**:
- The pitch shift ($7.87$ cents) is too small to leak significant energy in the STFT, keeping STFT similarity high.
- The noise ($0.0003$) is too quiet to trigger Cepstrum's log-amplitude instability, keeping Cepstrum similarity high.
- However, the combination of pitch shift ($7.87$ cents) and time shift ($13.8$ ms) translates and misaligns the autocorrelation peaks along the lag axis ($\tau$) across the entire 1-second lag space, leading to a near-total collapse (`0.0378`) of the rigid element-wise similarity.
