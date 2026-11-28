"""
Experiment 045 — The Key Claims on Real Data (issue #7)
=======================================================
Every earlier result rests on synthetic signals. This experiment re-runs the
key claims on public datasets with ground truth, using the *same analysis code*
as the synthetic experiments (Exp 042, 043, 044 analysis functions are imported,
not re-implemented), so any difference comes from the data.

Datasets (downloaded into data/raw/, see data/README.md):
  vocadito        40 solo singing excerpts, 7 languages, f0 annotated by musicians
                  (Bittner et al. 2021, CC BY 4.0)
  MDB-stem-synth  230 MedleyDB stems resynthesised so the f0 annotation is exact
                  (Salamon et al. 2017): real performances, many instruments
  BSDS500         500 photos with ~5 human boundary maps each (Arbeláez et al. 2011);
                  test split (200 images)

Claims tested:
  A1  Disagreement predicts pitch-lens blind spots (Exp 042)      ROC AUC per lens
  A2  The disagreement consensus beats single lenses and the
      engine-routed lens (Exp 042)                                 gross error rate
  A3  Lens failures are structured: ACF+Cepstrum fail together,
      STFT+Cepstrum rarely (Exp 043)                               co-failure lift
  A4  The framework's pitch tracker beats fixed YIN (Exp 039)     GER (20%), 50-cent GER
  I1  Disagreement predicts image-lens blind spots (Exp 044)       ROC AUC per lens
  I2  Image consensus vs single lenses (Exp 044)                   gross error rate
  I3  Image lenses fail in families (Exp 044)                      co-failure lift

Audio frames: 2048 samples at 22.05 kHz (as in Exp 042), non-overlapping. Truth =
geometric mean of the annotated f0 inside the frame; frames that are partly
voiced, span more than 100 cents, or lie outside the lenses' 80–1000 Hz range
are skipped. Fully unvoiced, non-silent frames are the "unpitched" check.
Conditions: the recording as is, white noise at 20 / 10 / 0 dB SNR (relative to
the track's mean power) and hard clipping (gain 10). MDB-stem-synth: at most 60
voiced frames per track (seeded), to keep tracks balanced.

Image patches: 64×64 grayscale patches from BSDS500 test images, centred on a
pixel marked by at least 3 annotators, kept only if every human boundary pixel
(any annotator) within 28 px lies within 4 px of one straight line through the
centre, and the grey levels on the two sides differ by at least 0.08. Truth =
that line's direction. At most 6 patches per image, at least 40 px apart.
Conditions: as is, noise σ 0.1 / 0.3, blur σ 2.5, JPEG-like q 20, 25% occlusion.

Cross-validation is grouped by track (audio) or image (photos), never by frame.

Usage:
  python src/experiments/exp045_real_data.py            # all (cached rows reused)
  python src/experiments/exp045_real_data.py --refresh  # recompute the cached rows

Outputs:
  results/exp045_real_data.png
  results/exp045_summary.json
"""

import glob
import json
import os
import pickle
import sys
import time
import warnings
import zlib

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in (project_root, os.path.join(project_root, "src"), os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.append(p)

warnings.filterwarnings("ignore")

import librosa  # noqa: E402
import scipy.io  # noqa: E402
from scipy import ndimage  # noqa: E402

import exp042_disagreement_prediction as e42  # noqa: E402
import exp043_disagreement_patterns as e43  # noqa: E402
import exp044_image_disagreement as e44  # noqa: E402
from src.framework.disagreement import DEFAULT_FMAX, DEFAULT_FMIN  # noqa: E402
from src.framework.engine import RepresentationIntelligenceEngine  # noqa: E402
from src.framework.plugins.pitch_tracker import AdaptivePitchTracker  # noqa: E402

SR = e42.SR
FRAME = e42.FRAME
RAW = os.path.join(project_root, "data", "raw")
CACHE = os.path.join(project_root, "data", "processed")
MAX_SPREAD_CENTS = 100.0
MDB_FRAMES_PER_TRACK = 60
TRACKER_SECONDS_MDB = 30
HOP = 512
LAG = 2   # frames of delay from the tracker's causal 5-frame median filter

OUT_PNG = os.path.join(project_root, "results", "exp045_real_data.png")
OUT_JSON = os.path.join(project_root, "results", "exp045_summary.json")

AUDIO_CONDITIONS = [
    ("clean", "-", lambda x, r: x),
    ("noise", "20 dB", lambda x, r: e42.add_noise_snr(x, 20, r)),
    ("noise", "10 dB", lambda x, r: e42.add_noise_snr(x, 10, r)),
    ("noise", "0 dB", lambda x, r: e42.add_noise_snr(x, 0, r)),
    ("clipping", "x10", lambda x, r: np.clip(10 * x / (np.max(np.abs(x)) + 1e-12), -1, 1)),
]

IMAGE_CONDITIONS = [
    ("clean", 0, lambda x, rng: x),
    ("noise", 0.1, lambda x, rng: x + rng.normal(0, 0.1, x.shape)),
    ("noise", 0.3, lambda x, rng: x + rng.normal(0, 0.3, x.shape)),
    ("blur", 2.5, lambda x, rng: ndimage.gaussian_filter(x, 2.5)),
    ("jpeg", 20, lambda x, rng: e44.jpeg_like(x, 20)),
    ("occlusion", 0.25, lambda x, rng: e44.occlude(x, 0.25, rng)),
]


def rng_for(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


# ──────────────────────────────────────────────────────────────────────────────
# AUDIO DATA
# ──────────────────────────────────────────────────────────────────────────────

def tracks_vocadito():
    root = os.path.join(RAW, "vocadito")
    for wav in sorted(glob.glob(os.path.join(root, "Audio", "*.wav"))):
        tid = os.path.basename(wav)[:-4]
        yield tid, wav, os.path.join(root, "Annotations", "F0", f"{tid}_f0.csv")


def tracks_mdb():
    root = os.path.join(RAW, "mdb_stem_synth", "MDB-stem-synth")
    for wav in sorted(glob.glob(os.path.join(root, "audio_stems", "*.wav"))):
        tid = os.path.basename(wav)[:-4]
        yield tid, wav, os.path.join(root, "annotation_stems", f"{tid}.csv")


DATASETS = {"vocadito": tracks_vocadito, "mdb_stem_synth": tracks_mdb}


def load_track(wav, ann):
    y, _ = librosa.load(wav, sr=SR, mono=True)
    a = np.loadtxt(ann, delimiter=",", ndmin=2)
    return y, a[:, 0], a[:, 1]


def frame_truth(times, f0, start):
    """Truth for the frame [start, start + FRAME): >0 voiced f0, 0.0 unvoiced, None skip."""
    m = (times >= start / SR) & (times < (start + FRAME) / SR)
    v = f0[m]
    if len(v) == 0:
        return None
    if np.all(v <= 0):
        return 0.0
    if np.any(v <= 0):
        return None
    if 1200 * np.log2(v.max() / v.min()) > MAX_SPREAD_CENTS:
        return None
    g = float(np.exp(np.mean(np.log(v))))
    return g if DEFAULT_FMIN <= g <= DEFAULT_FMAX else None


def collect_audio(name, max_frames=None):
    engine = RepresentationIntelligenceEngine()
    rows, unpitched = [], []
    for tid, wav, ann in DATASETS[name]():
        y, times, f0 = load_track(wav, ann)
        starts = np.arange(0, len(y) - FRAME, FRAME)
        truths = [(s, frame_truth(times, f0, s)) for s in starts]
        voiced = [(s, t) for s, t in truths if t]
        silent_floor = 1e-3 * np.max(np.abs(y))
        unvoiced = [(s, 0.0) for s, t in truths
                    if t == 0.0 and np.sqrt(np.mean(y[s:s + FRAME] ** 2)) > silent_floor]
        if max_frames:
            r = rng_for(name, tid, "subsample")
            if len(voiced) > max_frames:
                voiced = [voiced[i] for i in sorted(r.choice(len(voiced), max_frames, replace=False))]
            if len(unvoiced) > max_frames // 4:
                unvoiced = [unvoiced[i] for i in sorted(r.choice(len(unvoiced), max_frames // 4, replace=False))]
        for family, severity, fn in AUDIO_CONDITIONS:
            x = fn(y, rng_for(name, tid, family, severity))
            for s, t in voiced:
                rows.append(e42.frame_row(engine, x[s:s + FRAME], t, {
                    "dataset": name, "signal": name, "variant": tid, "family": family,
                    "severity": severity, "seed": 0, "group": tid, "f0": t}))
            if family == "clean":
                for s, _ in unvoiced:
                    unpitched.append(e42.frame_row(engine, x[s:s + FRAME], 0.0, {
                        "dataset": name, "group": tid, "seed": 0}))
    return rows, unpitched


def tracker_gain(name, seconds=None):
    """Exp 039 comparison on real audio: fixed YIN (2048, threshold 0.15) vs the
    framework's AdaptivePitchTracker, at hop 512, on annotated voiced frames."""
    engine = RepresentationIntelligenceEngine()
    out = {}
    for family, severity, fn in [AUDIO_CONDITIONS[0], AUDIO_CONDITIONS[2]]:
        est_b, est_a, gt, est_lag, gt_lag = [], [], [], [], []
        for tid, wav, ann in DATASETS[name]():
            y, times, f0 = load_track(wav, ann)
            if seconds:
                y = y[:seconds * SR]
            x = fn(y, rng_for(name, tid, family, severity, "tracker"))
            hops = np.arange(0, len(x) - HOP, HOP)
            if len(hops) == 0:
                continue
            base = librosa.yin(x, fmin=80, fmax=1000, sr=SR, frame_length=2048, hop_length=HOP,
                               trough_threshold=0.15, center=True)[:len(hops)]
            adap = AdaptivePitchTracker(engine).track(x, SR, hop_length=HOP)[:len(hops)]
            # truth at each hop: nearest annotation, voiced on both neighbours
            idx = np.clip(np.searchsorted(times, hops / SR), 1, len(times) - 1)
            truth = np.where((f0[idx] > 0) & (f0[idx - 1] > 0), f0[idx], 0.0)
            keep = (truth >= DEFAULT_FMIN) & (truth <= DEFAULT_FMAX)
            est_b.append(base[keep])
            est_a.append(adap[keep])
            gt.append(truth[keep])
            # The tracker's causal 5-frame median filter reports frame k - LAG at frame k.
            est_lag.append(adap[LAG:][keep[:-LAG]])
            gt_lag.append(truth[:-LAG][keep[:-LAG]])
        b, a, g = (np.concatenate(v) for v in (est_b, est_a, gt))
        a_lag, g_lag = np.concatenate(est_lag), np.concatenate(gt_lag)

        def rates(e, g=g):
            ok = e > 0
            rel = np.where(ok, np.abs(e - g) / g, np.inf)
            cents = np.where(ok, np.abs(1200 * np.log2(np.where(ok, e, 1.0) / g)), np.inf)
            return {"ger_20pct": float(np.mean(rel > 0.2)), "ger_50c": float(np.mean(cents > 50)),
                    "no_estimate": float(np.mean(~ok))}

        out[f"{family} {severity}"] = {"n_frames": int(len(g)), "fixed_yin": rates(b), "adaptive_tracker": rates(a),
                                       f"adaptive_tracker_lag{LAG}": rates(a_lag, g_lag)}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE DATA (BSDS500)
# ──────────────────────────────────────────────────────────────────────────────

BSDS = os.path.join(RAW, "bsds500", "BSR", "BSDS500", "data")
HALF = e44.N // 2
DISK = 28


def bsds_patches(split="test", per_image=6, min_dist=40, half=HALF):
    """Boundary patches of one BSDS500 split. ``half`` is half the side of the returned
    crop (the default gives the 64×64 patches of Exp 045; Exp 058 takes larger crops
    so it can rotate them and cut the centre back to 64×64)."""
    from PIL import Image

    yy, xx = np.mgrid[-DISK:DISK + 1, -DISK:DISK + 1]
    in_disk = np.hypot(yy, xx) <= DISK
    out = []
    for path in sorted(glob.glob(os.path.join(BSDS, "images", split, "*.jpg"))):
        iid = os.path.basename(path)[:-4]
        img = np.asarray(Image.open(path).convert("L"), float) / 255.0
        gts = scipy.io.loadmat(os.path.join(BSDS, "groundTruth", split, f"{iid}.mat"))["groundTruth"][0]
        bounds = [g["Boundaries"][0, 0].astype(bool) for g in gts]
        union = np.any(bounds, axis=0)
        support = sum(ndimage.binary_dilation(b) for b in bounds)
        H, W = img.shape
        cand = np.argwhere(union & (support >= 3))
        cand = cand[(cand[:, 0] >= half) & (cand[:, 0] < H - half) & (cand[:, 1] >= half) & (cand[:, 1] < W - half)]
        rng = rng_for("bsds", iid)
        rng.shuffle(cand)
        taken = []
        for r, c in cand:
            if len(taken) >= per_image:
                break
            if any(np.hypot(r - a, c - b) < min_dist for a, b in taken):
                continue
            win = union[r - DISK:r + DISK + 1, c - DISK:c + DISK + 1] & in_disk
            pts = np.argwhere(win).astype(float) - DISK          # (row, col) offsets
            if len(pts) < 30:
                continue
            mu = pts.mean(0)
            _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
            d_row, d_col = vt[0]
            normal = np.array([-d_col, d_row])                   # in (row, col)
            dist = (pts - mu) @ normal
            if np.max(np.abs(dist)) > 4.0 or abs(-mu @ normal) > 2.0:
                continue
            side = (np.stack([yy, xx], -1).astype(float) - mu) @ normal
            region = img[r - DISK:r + DISK + 1, c - DISK:c + DISK + 1]
            a, b = region[in_disk & (side > 3)], region[in_disk & (side < -3)]
            if len(a) < 50 or len(b) < 50 or abs(a.mean() - b.mean()) < 0.08:
                continue
            theta = float(np.degrees(np.arctan2(d_row, d_col)) % 180.0)  # (x = col, y = row)
            out.append((iid, theta, img[r - half:r + half, c - half:c + half].copy()))
            taken.append((r, c))
    return out


def collect_images(patches):
    analyzer = e44.make_image_analyzer()
    rows = []
    for k, (iid, theta, patch) in enumerate(patches):
        for family, severity, fn in IMAGE_CONDITIONS:
            x = fn(patch, rng_for("img", iid, k, family, severity))
            rows.append(e44.patch_row(analyzer, x, theta, {
                "variant": "bsds", "signal": "photo", "family": family, "severity": severity,
                "seed": 0, "group": iid}))
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────

def cached(name, fn, refresh):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"exp045_{name}.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    t = time.time()
    val = fn()
    with open(path, "wb") as f:
        pickle.dump(val, f)
    print(f"  {name}: computed in {time.time() - t:.0f} s")
    return val


def audio_summary(rows, unpitched):
    s42, _ = e42.analyze(rows)
    s42["unpitched"] = e42.analyze_unpitched(unpitched) if unpitched else None
    s43, _ = e43.analyze(rows + unpitched)
    keep = ("n_frames", "n_pitched", "k", "co_failure", "all_three_wrong", "state_map_comparison")
    s42["patterns"] = {k: s43[k] for k in keep}
    s42["patterns"]["named"] = [{k: p[k] for k in ("name", "share", "purity", "gross_error_rate")}
                                for p in s43["patterns"]]
    s42["n_tracks"] = len({r["group"] for r in rows})
    return s42


def group_bootstrap(groups, a_wrong, b_wrong, n_boot=2000, seed=0):
    """Difference in error rate (a − b), with a 95% CI from resampling whole groups
    (tracks or images), so frames from one recording are never treated as independent."""
    groups = np.asarray(groups)
    a, b = np.asarray(a_wrong, float), np.asarray(b_wrong, float)
    uniq, inv = np.unique(groups, return_inverse=True)
    sa, sb, n = (np.bincount(inv, weights=v, minlength=len(uniq)) for v in (a, b, np.ones_like(a)))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        k = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        diffs.append((k @ sa - k @ sb) / (k @ n))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(a.mean() - b.mean()), "ci95": [float(lo), float(hi)]}


def audio_significance(rows):
    g = [r["group"] for r in rows]
    wrong = {m: np.array([e42.cents_error(r["readings"][m][0], r["truth"]) > e42.GROSS_CENTS for r in rows])
             for m in e42.REPS}
    best = min(e42.REPS, key=lambda m: wrong[m].mean())
    routed = np.array([e42.cents_error(r["readings"][max(e42.REPS, key=lambda k: r["assumptions"][k])][0],
                                       r["truth"]) > e42.GROSS_CENTS for r in rows])
    cons = np.array([e42.cents_error(r["consensus_f0"], r["truth"]) > e42.GROSS_CENTS for r in rows])
    answered = np.array([r["consensus_f0"] > 0 for r in rows])
    return {
        "best_single_lens": best,
        "consensus_minus_best_lens": group_bootstrap(g, cons, wrong[best]),
        "consensus_minus_engine_routed": group_bootstrap(g, cons, routed),
        "consensus_when_answering_minus_best_lens_same_frames": group_bootstrap(
            np.array(g)[answered], cons[answered], wrong[best][answered]),
    }


def image_significance(all_rows):
    rows = [r for r in all_rows if not r["empty"]]
    g = np.array([r["group"] for r in rows])
    wrong = {m: np.array([e44.wrong(r, m) for r in rows]) for m in e44.REPS}
    best = min(e44.REPS, key=lambda m: wrong[m].mean())
    P = np.array([e42.grouped_cv_scores(np.array([e44.features(r, m) for r in rows]),
                                        wrong[m].astype(float), g) for m in e44.REPS]).T
    pick = np.array([e44.err_deg(r["readings"][e44.REPS[int(np.argmin(P[i]))]][0], r["truth"]) > e44.GROSS_DEG
                     for i, r in enumerate(rows)])
    cons = np.array([e44.err_deg(r["consensus"], r["truth"]) > e44.GROSS_DEG for r in rows])
    return {
        "best_single_lens": best,
        "learned_pick_minus_best_lens": group_bootstrap(g, pick, wrong[best]),
        "consensus_minus_best_lens": group_bootstrap(g, cons, wrong[best]),
    }


def main():
    refresh = "--refresh" in sys.argv
    summary = {"audio": {}, "tracker": {}, "image": None}
    for name, max_frames in (("vocadito", None), ("mdb_stem_synth", MDB_FRAMES_PER_TRACK)):
        if not any(True for _ in DATASETS[name]()):
            print(f"Skipping {name}: not found under data/raw (see data/README.md)")
            continue
        print(f"Audio: {name}")
        rows, unpitched = cached(f"audio_{name}", lambda: collect_audio(name, max_frames), refresh)
        summary["audio"][name] = audio_summary(rows, unpitched)
        summary["audio"][name]["significance"] = audio_significance(rows)
        seconds = TRACKER_SECONDS_MDB if name == "mdb_stem_synth" else None
        summary["tracker"][name] = cached(f"tracker_{name}", lambda: tracker_gain(name, seconds), refresh)

    if os.path.isdir(BSDS):
        print("Images: BSDS500")
        patches = cached("bsds_patches", bsds_patches, refresh)
        rows = cached("image_rows", lambda: collect_images(patches), refresh)
        s = e44.analyze(rows)
        s["n_base_patches"] = len(patches)
        s["n_images"] = len({p[0] for p in patches})
        s["significance"] = image_significance(rows)
        summary["image"] = s

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    plot(summary)
    report(summary)
    print(f"\nSaved {OUT_PNG}\nSaved {OUT_JSON}")


def report(s):
    for name, a in s["audio"].items():
        print(f"\n=== {name}: {a['n_frames']} frames from {a['n_tracks']} tracks ===")
        for rep in e42.REPS:
            p = a["per_rep"][rep]
            auc = p["auc"]
            print(f"  {rep:9s} gross {p['gross_error_rate']:6.1%}  AUC assumption {auc['assumption']:.3f}  "
                  f"state {auc['state_model']:.3f}  self {auc['self_conf']:.3f}  "
                  f"divergence {auc['divergence']:.3f}  disagree {auc['disagree_model']:.3f}")
        for m, v in a["fusion"].items():
            print(f"  {m:20s} gross {v['gross_error_rate']:6.1%}  abstain {v['abstain_rate']:6.1%}  "
                  f"when answering {v['gross_error_rate_when_answering']:6.1%}")
        for fam, v in a["by_family"].items():
            print(f"  family {fam:10s} gross {v['gross_error_rate']:6.1%}  AUC div {v['auc_divergence']:.3f}  "
                  f"assumption {v['auc_assumption']:.3f}")
        co = a["patterns"]["co_failure"]
        print("  co-failure lift:", {k: (round(v["lift"], 2) if v["lift"] else None) for k, v in co.items()})
        print("  NMI:", {k: round(v, 3) for k, v in a["patterns"]["state_map_comparison"].items()})
        print("  significance:", a["significance"])
        if a["unpitched"]:
            print("  unpitched:", a["unpitched"])
    for name, t in s["tracker"].items():
        print(f"\nTracker ({name}):")
        for cond, v in t.items():
            print(f"  {cond:10s} n={v['n_frames']:6d}  fixed YIN {v['fixed_yin']}  adaptive {v['adaptive_tracker']}  "
                  f"adaptive, lag-compensated {v[f'adaptive_tracker_lag{LAG}']}")
    im = s["image"]
    if im:
        print(f"\n=== BSDS500: {im['n_base_patches']} patches from {im['n_images']} images, "
              f"{im['n_analysed']} analysed rows ===")
        for rep in e44.REPS:
            p = im["prediction"][rep]
            print(f"  {rep:7s} gross {p['gross_error_rate']:6.1%}  self {p['auc_self_conf']:.3f}  "
                  f"divergence {p['auc_divergence']:.3f}  disagree {p['auc_disagree_model']:.3f}")
        for m, v in im["answer"].items():
            print(f"  {m:20s} gross {v['gross_error_rate']:6.1%}  abstain {v['abstain_rate']:6.1%}")
        print("  atlas:", {r: {k: round(v, 2) for k, v in im["atlas"][r].items()} for r in e44.REPS})
        print("  lift:", {k: round(v, 2) for k, v in im["co_failure_lift"].items()})
        print("  significance:", im["significance"])
        print("  lift within condition:", {k: round(v, 2) for k, v in im["co_failure_lift_within_condition"].items()})


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(17, 10.5))
    syn42 = json.load(open(os.path.join(project_root, "results", "exp042_summary.json")))
    syn44 = json.load(open(os.path.join(project_root, "results", "exp044_summary.json")))

    # Audio AUC: disagreement model vs own confidence vs assumption, synthetic vs real
    sets = [("synthetic (Exp 042)", syn42)] + [(n, a) for n, a in s["audio"].items()]
    keys = [("assumption", "engine assumption"), ("self_conf", "own confidence"), ("disagree_model", "disagreement model")]
    w = 0.8 / (len(sets) * len(keys))
    labels = []
    for i, (sname, summ) in enumerate(sets):
        for j, (k, kl) in enumerate(keys):
            off = (i * len(keys) + j) * w - 0.4 + w / 2
            vals = [summ["per_rep"][r]["auc"][k] for r in e42.REPS]
            ax[0, 0].bar(np.arange(3) + off, vals, w, color=f"C{j}", alpha=[0.35, 0.7, 1.0][min(i, 2)],
                         label=f"{kl}" if i == len(sets) - 1 else None)
        labels.append(sname)
    ax[0, 0].set_xticks(range(3), e42.REPS)
    ax[0, 0].axhline(0.5, color="grey", ls="--", lw=0.8)
    ax[0, 0].set_ylim(0.3, 1.0)
    ax[0, 0].set_ylabel("ROC AUC")
    ax[0, 0].set_title("A1. Predicting pitch-lens errors. Bars per lens: " + " → ".join(labels), fontsize=9)
    ax[0, 0].legend(fontsize=8, loc="lower left")

    methods = ["stft", "acf", "cepstrum", "engine_routed", "consensus"]
    w = 0.8 / len(sets)
    for i, (sname, summ) in enumerate(sets):
        vals = [summ["fusion"][m]["gross_error_rate"] for m in methods]
        ax[0, 1].bar(np.arange(len(methods)) + i * w - 0.4 + w / 2, vals, w, label=sname)
    ax[0, 1].set_xticks(range(len(methods)), [m.replace("_", "\n") for m in methods])
    ax[0, 1].set_ylabel("gross error rate (> 50 cents; abstaining counts as wrong)")
    ax[0, 1].set_title("A2. Is the consensus a better answer? (audio)")
    ax[0, 1].legend(fontsize=8)

    if s["image"]:
        im = s["image"]
        keys = [("auc_self_conf", "own confidence"), ("auc_divergence", "divergence"),
                ("auc_disagree_model", "disagreement model")]
        w = 0.13
        for i, (sname, summ) in enumerate([("synthetic (Exp 044)", syn44), ("BSDS500", im)]):
            for j, (k, kl) in enumerate(keys):
                off = (i * 3 + j) * w - 0.39 + w / 2
                ax[1, 0].bar(np.arange(5) + off, [summ["prediction"][r][k] for r in e44.REPS], w,
                             color=f"C{j}", alpha=0.45 if i == 0 else 1.0, label=kl if i == 1 else None)
        ax[1, 0].set_xticks(range(5), e44.REPS)
        ax[1, 0].axhline(0.5, color="grey", ls="--", lw=0.8)
        ax[1, 0].set_ylim(0.3, 1.0)
        ax[1, 0].set_ylabel("ROC AUC")
        ax[1, 0].set_title("I1. Predicting image-lens errors (light = synthetic, dark = BSDS500)")
        ax[1, 0].legend(fontsize=8, loc="lower left")

        methods = [*e44.REPS, "most_confident", "consensus", "learned_pick", "oracle_best_single"]
        for i, (sname, summ) in enumerate([("synthetic (Exp 044)", syn44), ("BSDS500", im)]):
            ax[1, 1].bar(np.arange(len(methods)) + (i - 0.5) * 0.38, [summ["answer"][m]["gross_error_rate"]
                                                                      for m in methods], 0.38, label=sname)
        ax[1, 1].set_xticks(range(len(methods)), [m.replace("_", "\n") for m in methods], fontsize=8)
        ax[1, 1].set_ylabel("gross error rate (> 10°)")
        ax[1, 1].set_title("I2. Is the consensus a better answer? (images)")
        ax[1, 1].legend(fontsize=8)

    fig.suptitle("Exp 045 — The key claims on real data (vocadito, MDB-stem-synth, BSDS500)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)


if __name__ == "__main__":
    main()
