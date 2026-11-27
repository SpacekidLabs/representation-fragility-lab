"""
Adaptive Pitch Tracking Plugin
==============================
Implements the AdaptivePitchTracker which dynamically adapts window size,
trough threshold, and voicing/hold gating flags using YIN under the engine.
"""

import numpy as np
import scipy.signal
import librosa
from ..engine import RepresentationIntelligenceEngine

# Consecutive, mutually consistent raw estimates (~70 ms at hop 512 / 22.05 kHz)
# needed to override the continuity / octave constraints.
REACQUIRE_FRAMES = 3

class AdaptivePitchTracker:
    """
    State-Space Adaptive YIN Pitch Tracker.
    Queries the RepresentationIntelligenceEngine frame-by-frame to scale the
    YIN window size and trough threshold, and apply context-aware hold logic.
    """
    def __init__(self, engine=None):
        self.engine = engine or RepresentationIntelligenceEngine()

    def track(self, sig: np.ndarray, sr: int, hop_length=512, fmin=80, fmax=1000) -> np.ndarray:
        """
        Track the pitch of the input signal.
        
        Parameters:
            sig: 1D NumPy array of time-domain audio.
            sr: sample rate.
            hop_length: frame hop size (default 512).
            fmin: minimum frequency in Hz (default 80).
            fmax: maximum frequency in Hz (default 1000).
            
        Returns:
            np.ndarray: 1D array of estimated pitch frequencies (Hz) per frame.
        """
        num_samples = len(sig)
        pad_len = 2048
        y_pad = np.pad(sig, pad_len, mode='reflect')
        
        hop_indices = np.arange(0, num_samples - hop_length, hop_length)
        N_frames = len(hop_indices)
        
        pitches = np.zeros(N_frames)
        # No made-up starting anchor: continuity constraints only apply once a
        # real pitch has been tracked (a fixed 220 Hz anchor kept the tracker
        # stuck whenever the signal started more than 3 semitones away; #10).
        last_valid_pitch = 0.0
        pitch_history = [0.0] * 5
        unvoiced_consecutive_frames = 0
        override_run = []  # consecutive raw estimates overridden by Step 1
        
        for idx, hop_idx in enumerate(hop_indices):
            center_idx = hop_idx
            
            # Analyze state using a standard local 1024-sample window centered at hop index
            f_win = 1024
            f_start = center_idx + pad_len - f_win // 2
            f_end = center_idx + pad_len + f_win // 2
            f_frame = y_pad[f_start:f_end]
            
            # Downsample to 22.05kHz for engine analyze consistency
            if sr != 22050:
                f_frame_22k = scipy.signal.resample(f_frame, f_win // 2)
                state = self.engine.analyze(f_frame_22k, 22050, compute_disagreement=False)
            else:
                state = self.engine.analyze(f_frame, sr, compute_disagreement=False)
                
            # Retrieve recommendations from engine
            win_size = state.recommended_window
            params = state.recommended_parameters["pitch_tracking"]
            trough_thresh = params["yin_trough"]
            voicing_gate = params["voicing_gate"]
            hold_pitch = params["hold_pitch"]
            
            # Extract the dynamically scaled window centered at the hop index
            a_start = center_idx + pad_len - win_size // 2
            a_end = center_idx + pad_len + win_size // 2
            a_frame = y_pad[a_start:a_end]
            
            # Run YIN on the adaptive frame size
            try:
                pitch_array = librosa.yin(
                    a_frame, fmin=fmin, fmax=fmax, sr=sr, 
                    frame_length=win_size, hop_length=win_size, 
                    trough_threshold=trough_thresh, center=False
                )
                pitch = float(pitch_array[0])
            except Exception:
                pitch = 0.0

            # Step 1: Octave correction and continuity constraint relative to last_valid_pitch
            raw_pitch = pitch
            if last_valid_pitch > 0.0 and pitch > 0.0:
                ratio = pitch / last_valid_pitch
                
                # Detect and correct octave tracking errors
                if abs(ratio - 0.5) < 0.08:
                    pitch *= 2.0
                    ratio *= 2.0
                elif abs(ratio - 2.0) < 0.30:
                    pitch *= 0.5
                    ratio *= 0.5
                elif abs(ratio - 0.25) < 0.04:
                    pitch *= 4.0
                    ratio *= 4.0
                elif abs(ratio - 4.0) < 0.60:
                    pitch *= 0.25
                    ratio *= 0.25
                
                # Enforce absolute continuity limit: max 3.0 semitones change per frame
                max_ratio = 2**(3/12)
                min_ratio = 2**(-3/12)
                if ratio > max_ratio or ratio < min_ratio:
                    pitch = last_valid_pitch

                # Re-acquisition: if Step 1 keeps overriding the raw estimate and
                # the raw estimates agree with each other, the pitch really moved
                # (note change, octave leap, new phrase). Accept it instead of
                # holding the old pitch forever.
                if abs(12.0 * np.log2(pitch / raw_pitch)) > 1.0:
                    if override_run and abs(12.0 * np.log2(raw_pitch / override_run[-1])) < 1.0:
                        override_run.append(raw_pitch)
                    else:
                        override_run = [raw_pitch]
                    if len(override_run) >= REACQUIRE_FRAMES:
                        pitch = raw_pitch
                        pitch_history = [0.0] * 5  # restart the median filter at the new pitch
                        override_run = []
                else:
                    override_run = []

            # Step 2: Determine raw voicing state
            raw_is_voiced = (pitch > fmin and pitch < fmax)
            
            # Apply Engine recommendations and confidence constraints
            acf_conf = state.assumptions.get("acf", 1.0)
            if voicing_gate and acf_conf < 0.35:
                raw_is_voiced = False
            if hold_pitch and acf_conf < 0.20:
                pitch = last_valid_pitch
                raw_is_voiced = (pitch > fmin and pitch < fmax)

            # Step 3: Run voiced-only median filter
            if raw_is_voiced:
                # Initialize history if empty to avoid onset lag
                if all(val == 0.0 for val in pitch_history):
                    pitch_history = [pitch] * 5
                else:
                    pitch_history.pop(0)
                    pitch_history.append(pitch)
                
                # Set pitch to median of history
                pitch = float(np.median(pitch_history))
                last_valid_pitch = pitch
            else:
                pitch = 0.0

            # Step 4: Apply release gate hysteresis (hold last pitch for up to 3 frames ~35 ms)
            is_voiced = raw_is_voiced
            if raw_is_voiced:
                unvoiced_consecutive_frames = 0
            else:
                unvoiced_consecutive_frames += 1
                if unvoiced_consecutive_frames < 3 and last_valid_pitch > fmin and last_valid_pitch < fmax:
                    is_voiced = True
                    pitch = last_valid_pitch
                else:
                    # Full release: reset the median filter and forget the anchor,
                    # so the next phrase is tracked from scratch.
                    pitch_history = [0.0] * 5
                    last_valid_pitch = 0.0
                    override_run = []
            
            pitches[idx] = pitch if is_voiced else 0.0
            
        return pitches


class QuorumPitchTracker:
    """
    Pitch tracker from Exp 054 (issue #28): the Exp 046 quorum of the plain lenses
    (STFT, ACF, Cepstrum on a 2048-sample frame centred on each hop; consensus
    only when at least two trusted lenses agree), falling back to fixed YIN
    (2048 samples, trough threshold 0.15) when there is no quorum.

    No median filter, continuity limit, octave correction or engine gates: on
    vocadito and MDB-stem-synth the 5-hop causal median of AdaptivePitchTracker
    (a 2-hop delay) caused most of its loss, and the other steps changed the
    error by less than 0.5 pts. It beats fixed YIN on both datasets, clean and
    with 10 dB noise. Like YIN it reports a pitch on every hop; it does not
    decide voicing.
    """

    def __init__(self, min_agreeing: int = 2):
        self.min_agreeing = min_agreeing

    def track(self, sig: np.ndarray, sr: int, hop_length=512, fmin=80, fmax=1000) -> np.ndarray:
        from ..audio import PitchComparator
        from ..disagreement import OUTLIER_MARGIN, VOICING_THRESHOLD, read_acf, read_cepstrum, read_stft
        from ..audio import ConfidenceThresholdDetector, Reading

        readers = {"stft": read_stft, "acf": read_acf, "cepstrum": read_cepstrum}
        comp = PitchComparator(outlier_margin=OUTLIER_MARGIN, min_agreeing=self.min_agreeing,
                               detector=ConfidenceThresholdDetector(VOICING_THRESHOLD))
        n = len(sig)
        hops = np.arange(0, n - hop_length, hop_length)
        yin = librosa.yin(sig, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=hop_length,
                          trough_threshold=0.15, center=True)[:len(hops)]
        pad = 2048
        y = np.pad(sig, pad, mode="reflect")
        out = np.zeros(len(hops))
        for i, c in enumerate(hops):
            frame = y[c + pad - 1024:c + pad + 1024]
            rd = {}
            for name, read in readers.items():
                r = read(frame, sr, fmin, fmax)
                conf = r.confidence if np.isfinite(r.confidence) else 0.0
                rd[name] = Reading(name, r.f0 if r.f0 > 0 else None, conf)
            q = comp.compare(rd).consensus
            p = q if q is not None else float(yin[i])
            out[i] = p if fmin < p < fmax else 0.0
        return out
