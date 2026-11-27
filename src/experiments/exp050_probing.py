"""
Experiment 050: perturb the signal to settle a disagreement (issue #24)
=======================================================================
When the pitch lenses disagree, every fusion rule so far only *voted* on the
readings it had. Here we ask the signal a question instead: apply a small,
known change and see which candidate answer moves the way it should.

Candidates. The distinct answers of STFT, ACF and Cepstrum on a frame (lens
estimates within 50 cents are merged into one candidate, their geometric
mean). A frame is a *disagreement frame* when it has two or more candidates.
On those frames one candidate is right 98-99% of the time, so choosing well is
worth a lot.

Probes. For a candidate c, each probe predicts what the lenses should read if
c is the true f0; a lens *follows* the probe when it reads within 50 cents of
the prediction.
  none      the frame as is (prediction c): this is plain voting
  shift     resample the frame by 2 semitones (up, or down if c * r > 1 kHz
            for some candidate): prediction c * r. The change is exact for
            every component, so a lens that tracks the true period follows it.
  noise     add white noise 10 dB below the frame power: prediction c
  lowpass   5th-order low-pass at 1.5 c: prediction c. If c is right the
            fundamental survives and every lens should find it; if the true f0
            is 2c nothing pitched is left; if it is c / 2 the lenses go there.
  highpass  5th-order high-pass at 1.5 c: prediction c. If c is right the
            fundamental is gone and the periodicity lenses (ACF, Cepstrum)
            should still read c (the missing-fundamental blind spot of
            Exp 008 says the STFT will not); if the truth is c / 2 they read c / 2.
shift and noise cost one extra set of readings per frame; lowpass and highpass
one per candidate.

Choosers, all picking the best-scoring candidate (ties: more lenses at
baseline, then higher confidence)
  count     number of (probe, lens) follows, no training
  prior     naive Bayes: log-likelihood ratio of each (probe, lens) follow,
            P(follow | c right) / P(follow | c wrong), estimated on the
            synthetic frames of Exp 042. This is the blind-spot map used as the
            prior on how each lens should react; it is applied unchanged to
            the real recordings.
  prior_vote   the same, using only the "none" probe (no perturbation): what
               the prior gives without probing
  learned   logistic regression on the 15 follow bits, 5-fold cross-validation
            split by recording (within each dataset)

Baselines on the same frames: legacy consensus (fuse all trusted), quorum 2
(abstention counts as wrong), the dataset's best single lens (chosen in
hindsight) and the candidate oracle.

Datasets: the synthetic frames of Exp 042 and the clean / noisy / clipped
frames of Exp 045 (vocadito, MDB-stem-synth). The frames are rebuilt with the
same code and seeds as those experiments and checked against their cached
readings. MDB has 24 668 disagreement frames; a seeded sample of 6000 is probed.

Audio only. What would carry over to images is noted in docs/EXPERIMENTS.md
(rotation is the analogue of the pitch shift, blur of the low-pass).

Run:  python src/experiments/exp050_probing.py   (uses data/processed caches of 045/046)
Outputs: results/exp050_probing.png, results/exp050_summary.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import exp042_disagreement_prediction as e42  # noqa: E402
import exp045_real_data as e45  # noqa: E402
import exp046_quorum as e46  # noqa: E402
from framework.disagreement import DEFAULT_FMAX, read_acf, read_cepstrum, read_stft  # noqa: E402
from perturbations.filter import lowpass_filter  # noqa: E402

CACHE = os.path.join(ROOT, "data", "processed")
OUT_JSON = os.path.join(ROOT, "results", "exp050_summary.json")
OUT_PNG = os.path.join(ROOT, "results", "exp050_probing.png")
SR, FRAME, GROSS = e42.SR, e42.FRAME, e42.GROSS_CENTS
LENSES = ("stft", "acf", "cepstrum")
READERS = {"stft": read_stft, "acf": read_acf, "cepstrum": read_cepstrum}
PROBES = ("none", "shift", "noise", "lowpass", "highpass")
SHIFT = 2.0 ** (2 / 12)
CTX = int(np.ceil(FRAME * SHIFT)) + 4
NOISE_DB = 10.0
MAX_PROBED = 6000
FOLDS = 5
DATASETS = {"synthetic": "exp046_e42_rows", "vocadito": "exp045_audio_vocadito",
            "mdb_stem_synth": "exp045_audio_mdb_stem_synth"}


def cents(a, b):
    return abs(1200.0 * np.log2(a / b)) if a > 0 and b > 0 else np.inf


# ──────────────────────────────────────────────────────────────────────────────
# Frames, rebuilt in the order of the cached rows
# ──────────────────────────────────────────────────────────────────────────────

def synthetic_contexts(wanted):
    """Exp 042's collect(), yielding (row index, CTX samples from the frame start)."""
    i = 0
    for f0 in e42.F0S:
        for variant, _, make in e42.signal_variants(f0):
            for family, severity, fn, track_fn in e42.conditions(f0):
                for seed in e42.SEEDS:
                    if i in wanted:
                        rng = np.random.default_rng(zlib.crc32(repr((f0, variant, family, severity, seed)).encode()))
                        x, track = make(rng)
                        x = fn(x, rng)
                        start = (len(x) - FRAME) // 2 + int(rng.integers(-500, 500))
                        yield i, x[start:start + CTX]
                    i += 1


def real_contexts(name, wanted, max_frames):
    """Exp 045's collect_audio(), voiced rows only, same order."""
    i = 0
    for tid, wav, ann in e45.DATASETS[name]():
        y, times, f0 = e45.load_track(wav, ann)
        starts = np.arange(0, len(y) - FRAME, FRAME)
        voiced = [(s, t) for s, t in ((s, e45.frame_truth(times, f0, s)) for s in starts) if t]
        if max_frames:
            r = e45.rng_for(name, tid, "subsample")
            if len(voiced) > max_frames:
                voiced = [voiced[k] for k in sorted(r.choice(len(voiced), max_frames, replace=False))]
        for family, severity, fn in e45.AUDIO_CONDITIONS:
            n = len(voiced)
            if not any(j in wanted for j in range(i, i + n)):
                i += n
                continue
            x = fn(y, e45.rng_for(name, tid, family, severity))
            for s, _ in voiced:
                if i in wanted:
                    ctx = x[s:s + CTX]
                    yield i, np.pad(ctx, (0, CTX - len(ctx)))
                i += 1


def unpitched_contexts(name):
    """Unpitched frames of Exp 042 (white noise) or Exp 045 (unvoiced, non-silent, clean)."""
    if name == "synthetic":
        for seed in range(e42.UNPITCHED_FRAMES):
            frame = np.random.default_rng(10_000 + seed).normal(0.0, 0.3, FRAME)
            tail = np.random.default_rng(20_000 + seed).normal(0.0, 0.3, CTX - FRAME)
            yield seed, np.r_[frame, tail]
        return
    max_frames = e45.MDB_FRAMES_PER_TRACK if name == "mdb_stem_synth" else None
    i = 0
    for tid, wav, ann in e45.DATASETS[name]():
        y, times, f0 = e45.load_track(wav, ann)
        starts = np.arange(0, len(y) - FRAME, FRAME)
        truths = [(s, e45.frame_truth(times, f0, s)) for s in starts]
        floor = 1e-3 * np.max(np.abs(y))
        unvoiced = [s for s, t in truths if t == 0.0 and np.sqrt(np.mean(y[s:s + FRAME] ** 2)) > floor]
        if max_frames:
            r = e45.rng_for(name, tid, "subsample")
            voiced = [s for s, t in truths if t]
            if len(voiced) > max_frames:  # consume the same random draws as Exp 045
                r.choice(len(voiced), max_frames, replace=False)
            if len(unvoiced) > max_frames // 4:
                unvoiced = [unvoiced[k] for k in sorted(r.choice(len(unvoiced), max_frames // 4, replace=False))]
        for s in unvoiced:
            ctx = y[s:s + CTX]
            yield i, np.pad(ctx, (0, CTX - len(ctx)))
            i += 1


# ──────────────────────────────────────────────────────────────────────────────
# Candidates and probes
# ──────────────────────────────────────────────────────────────────────────────

def candidates(readings):
    """Distinct answers: lens estimates within 50 cents merged (geometric mean)."""
    est = sorted((f, n) for n, (f, _) in readings.items() if f > 0)
    groups = []
    for f, n in est:
        if groups and cents(f, groups[-1][-1][0]) <= GROSS:
            groups[-1].append((f, n))
        else:
            groups.append([(f, n)])
    return [(float(np.exp(np.mean(np.log([f for f, _ in g])))), tuple(n for _, n in g)) for g in groups]


def read3(frame):
    out = {}
    for n in LENSES:
        r = READERS[n](frame, SR)
        out[n] = (float(r.f0), float(r.confidence))
    return out


def highpass(x, cutoff):
    return e42.highpass(x, cutoff) if cutoff < 0.5 * SR else np.zeros_like(x)


def shifted(ctx, ratio):
    """Resample so that every frequency is multiplied by ``ratio``."""
    t = np.arange(FRAME) * ratio
    return np.interp(t, np.arange(len(ctx)), ctx)


def probe(ctx, rng):
    """All probe readings for one frame. Returns the baseline readings, the candidates,
    follow bits [candidate, probe, lens], the number of extra lens reads and the times."""
    frame = ctx[:FRAME]
    t0 = time.perf_counter()
    base = read3(frame)
    t_base = time.perf_counter() - t0
    cands = candidates(base)
    t0 = time.perf_counter()
    ratio = SHIFT if max(c for c, _ in cands) * SHIFT <= DEFAULT_FMAX else 1.0 / SHIFT
    shared = {"shift": read3(shifted(ctx, ratio))}
    p = np.mean(frame ** 2)
    shared["noise"] = read3(frame + rng.normal(0.0, np.sqrt(p / 10 ** (NOISE_DB / 10)), FRAME))
    reads = 2
    bits = np.zeros((len(cands), len(PROBES), len(LENSES)), bool)
    for k, (c, _) in enumerate(cands):
        per = {"none": (base, c), "shift": (shared["shift"], c * ratio), "noise": (shared["noise"], c),
               "lowpass": (read3(lowpass_filter(frame, SR, 1.5 * c)), c),
               "highpass": (read3(highpass(frame, 1.5 * c)), c)}
        reads += 2
        for j, pr in enumerate(PROBES):
            rd, pred = per[pr]
            for m, n in enumerate(LENSES):
                bits[k, j, m] = cents(rd[n][0], pred) <= GROSS
    return {"base": base, "cands": cands, "bits": bits, "ratio": ratio, "extra_reads": 3 * reads,
            "t_base": t_base, "t_probe": time.perf_counter() - t0}


def probe_dataset(name, refresh=False):
    path = os.path.join(CACHE, f"exp050_probes_{name}.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    rows = e46.cached(DATASETS[name], None)
    rows = rows[0] if isinstance(rows, tuple) else rows
    dis = [i for i, r in enumerate(rows) if len(candidates(r["readings"])) >= 2]
    if len(dis) > MAX_PROBED:
        dis = sorted(np.random.default_rng(50).choice(dis, MAX_PROBED, replace=False).tolist())
    wanted = set(dis)
    gen = (synthetic_contexts(wanted) if name == "synthetic"
           else real_contexts(name, wanted, e45.MDB_FRAMES_PER_TRACK if name == "mdb_stem_synth" else None))
    out, mismatch, t = {}, 0, time.time()
    for i, ctx in gen:
        res = probe(ctx, np.random.default_rng(zlib.crc32(repr((name, i, "probe")).encode())))
        if any(cents(res["base"][n][0], rows[i]["readings"][n][0]) > 1.0 for n in LENSES
               if rows[i]["readings"][n][0] > 0):
            mismatch += 1
        out[i] = res
        if len(out) % 1000 == 0:
            print(f"    {name}: {len(out)}/{len(dis)} frames probed ({time.time() - t:.0f} s)")
    result = {"probes": out, "mismatch": mismatch, "n_disagree_total": sum(
        len(candidates(r["readings"])) >= 2 for r in rows)}
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def probe_unpitched(name, refresh=False):
    """Probe every unpitched frame that has two or more candidates (those are the
    ones a prober would be asked to settle)."""
    path = os.path.join(CACHE, f"exp050_unpitched_{name}.pkl")
    if os.path.exists(path) and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    nulls = e46.cached(DATASETS[name].replace("rows", "unpitched") if name == "synthetic" else DATASETS[name], None)
    nulls = nulls[1] if isinstance(nulls, tuple) else nulls
    out, mismatch, n = {}, 0, 0
    for i, ctx in unpitched_contexts(name):
        n += 1
        res = probe(ctx, np.random.default_rng(zlib.crc32(repr((name, i, "null")).encode())))
        if any(cents(res["base"][m][0], nulls[i]["readings"][m][0]) > 1.0 for m in LENSES
               if nulls[i]["readings"][m][0] > 0):
            mismatch += 1
        if len(res["cands"]) >= 2:
            out[i] = res
    result = {"probes": out, "n": n, "n_cached": len(nulls), "mismatch": mismatch}
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Choosers
# ──────────────────────────────────────────────────────────────────────────────

def tiebreak(res, k):
    c, members = res["cands"][k]
    return (len(members), max(res["base"][n][1] for n in members))


def pick(res, scores):
    return max(range(len(scores)), key=lambda k: (scores[k],) + tiebreak(res, k))


def labels(res, truth):
    return np.array([cents(c, truth) <= GROSS for c, _ in res["cands"]])


def fit_prior(examples, probes=PROBES, alpha=1.0):
    """examples: (bits [cand, probe, lens], right [cand]) pairs. Log-likelihood ratios."""
    idx = [PROBES.index(p) for p in probes]
    B = np.concatenate([b[:, idx, :] for b, _ in examples]).reshape(-1, len(idx) * len(LENSES)).astype(float)
    y = np.concatenate([r for _, r in examples])
    p1 = (B[y].sum(0) + alpha) / (y.sum() + 2 * alpha)
    p0 = (B[~y].sum(0) + alpha) / ((~y).sum() + 2 * alpha)
    w1, w0 = np.log(p1 / p0), np.log((1 - p1) / (1 - p0))
    table = {f"{probes[j // 3]}:{LENSES[j % 3]}": [float(p1[j]), float(p0[j])] for j in range(len(p1))}

    def score(bits):
        b = bits[:, idx, :].reshape(len(bits), -1)
        return b @ w1 + (1 - b) @ w0
    return score, table


def fit_learned(examples):
    X = np.concatenate([b.reshape(len(b), -1) for b, _ in examples]).astype(float)
    y = np.concatenate([r for _, r in examples]).astype(float)
    f = e42.fit_logistic(X, y)
    return lambda bits: f(bits.reshape(len(bits), -1).astype(float))


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(name, rows, probed, prior, prior_vote):
    idx = sorted(probed)
    sub = [rows[i] for i in idx]
    truth = np.array([r["truth"] for r in sub])
    groups = np.array([str(r["group"]) for r in sub])
    res = [probed[i] for i in idx]
    right = [labels(p, t) for p, t in zip(res, truth)]

    def wrong_of(est):
        return np.array([cents(e, t) > GROSS for e, t in zip(est, truth)])

    wrong = {}
    for rule, k in (("legacy", None), ("q2", 2)):
        comp = e46.audio_comparator(k)
        est = [e46.audio_consensus(r, comp) for r in sub]
        wrong[rule] = wrong_of(est)
        if rule == "q2":
            q2_ans = np.array([e > 0 for e in est])
    lens_wrong = {n: wrong_of([r["readings"][n][0] for r in sub]) for n in LENSES}
    best = min(lens_wrong, key=lambda n: lens_wrong[n].mean())
    wrong["best_lens"] = lens_wrong[best]
    wrong["oracle"] = np.array([not r.any() for r in right])

    def choose(score_fn):
        return np.array([not right[j][pick(p, list(score_fn(p["bits"])))] for j, p in enumerate(res)])

    wrong["count"] = choose(lambda b: b.sum((1, 2)))
    lp = PROBES.index("lowpass")
    wrong["lowpass_count"] = choose(lambda b: b[:, lp, :].sum(1))
    wrong["prior_vote"] = choose(prior_vote)
    wrong["prior"] = choose(prior)
    # learned: grouped CV inside the dataset
    uniq = np.unique(groups)
    np.random.default_rng(0).shuffle(uniq)
    learned = np.zeros(len(res), bool)
    for fold in np.array_split(uniq, FOLDS):
        test = np.isin(groups, fold)
        f = fit_learned([(res[j]["bits"], right[j]) for j in np.where(~test)[0]])
        for j in np.where(test)[0]:
            learned[j] = not right[j][pick(res[j], list(f(res[j]["bits"])))]
    wrong["learned"] = learned

    t_base = np.array([p["t_base"] for p in res])
    t_probe = np.array([p["t_probe"] for p in res])
    reads = np.array([p["extra_reads"] for p in res])
    families = list(dict.fromkeys(r["family"] for r in sub))
    out = {
        "n_probed": len(idx),
        "n_candidates_mean": float(np.mean([len(p["cands"]) for p in res])),
        "best_lens": best,
        "gross": {k: float(v.mean()) for k, v in wrong.items()},
        "q2_answer_rate": float(q2_ans.mean()),
        "q2_gross_answered": float(wrong["q2"][q2_ans].mean()),
        "on_q2_answered": {k: float(v[q2_ans].mean()) for k, v in wrong.items()},
        "by_family": {f: {k: float(v[[r["family"] == f for r in sub]].mean())
                          for k, v in wrong.items()} for f in families},
        "vs_q2": {k: e45.group_bootstrap(groups, wrong[k], wrong["q2"])
                  for k in ("count", "lowpass_count", "prior", "learned", "prior_vote")},
        "vs_legacy": {k: e45.group_bootstrap(groups, wrong[k], wrong["legacy"])
                      for k in ("count", "lowpass_count", "prior", "learned", "prior_vote")},
        "vs_best_lens": {k: e45.group_bootstrap(groups, wrong[k], wrong["best_lens"])
                         for k in ("count", "lowpass_count", "learned")},
        "cost": {"extra_lens_reads_per_frame": float(reads.mean()),
                 "lowpass_only_extra_reads_per_frame": float(np.mean([3 * len(p["cands"]) for p in res])),
                 "time_ratio_vs_base_reads": float(t_probe.sum() / t_base.sum()),
                 "ms_per_frame_base": float(1e3 * t_base.mean()),
                 "ms_per_frame_probe": float(1e3 * t_probe.mean())},
    }
    # how much of the oracle gap is closed relative to q2 and legacy
    for ref in ("legacy", "q2"):
        gap = out["gross"][ref] - out["gross"]["oracle"]
        out[f"gap_closed_vs_{ref}"] = {k: float((out["gross"][ref] - out["gross"][k]) / gap) if gap > 0 else float("nan")
                                       for k in ("count", "lowpass_count", "prior", "learned")}
    return out, wrong


def unpitched_check(rows, probed, nulls, nprobed):
    """Probing always answers. How separable are unpitched frames by the best candidate's
    follow count (the untrained 'count' score), and what would an abstain threshold cost?"""
    pitched = [probed[i]["bits"].sum((1, 2)).max() for i in sorted(probed)]
    right = [labels(probed[i], rows[i]["truth"])[pick(probed[i], list(probed[i]["bits"].sum((1, 2))))]
             for i in sorted(probed)]
    null = [p["bits"].sum((1, 2)).max() for p in nulls["probes"].values()]
    out = {"n_unpitched": nulls["n"], "n_unpitched_cached": nulls["n_cached"],
           "rebuild_mismatch": nulls["mismatch"],
           "frac_unpitched_with_2plus_candidates": len(null) / max(nulls["n"], 1),
           "auc_best_count_pitched_vs_unpitched": e42.roc_auc(np.r_[pitched, null], np.r_[np.ones(len(pitched)), np.zeros(len(null))]) if null else float("nan"),
           "thresholds": {}}
    pitched, right, null = np.array(pitched), np.array(right), np.array(null)
    for t in range(0, 16):
        ans = pitched >= t
        out["thresholds"][t] = {"answer_pitched": float(ans.mean()),
                                "gross_all_pitched": float(np.mean(~(ans & right))),
                                "false_answer_unpitched_2plus": float(np.mean(null >= t)) if len(null) else float("nan")}
    return out


def follow_rates(rows, probed):
    """P(follow | candidate right / wrong) per probe and lens: the measured reaction map."""
    hit = {True: [], False: []}
    for i, p in probed.items():
        for b, r in zip(p["bits"], labels(p, rows[i]["truth"])):
            hit[bool(r)].append(b)
    return {f"{pr}:{n}": {"right": float(np.mean([b[j, m] for b in hit[True]])),
                          "wrong": float(np.mean([b[j, m] for b in hit[False]]))}
            for j, pr in enumerate(PROBES) for m, n in enumerate(LENSES)}


def main():
    refresh = "--refresh" in sys.argv
    data, nulls = {}, {}
    for name in DATASETS:
        rows = e46.cached(DATASETS[name], None)
        rows = rows[0] if isinstance(rows, tuple) else rows
        print(f"Probing {name}")
        data[name] = (rows, probe_dataset(name, refresh))
        nulls[name] = probe_unpitched(name, refresh)

    syn_rows, syn = data["synthetic"]
    syn_examples = [(p["bits"], labels(p, syn_rows[i]["truth"])) for i, p in syn["probes"].items()]
    prior, table = fit_prior(syn_examples)
    prior_vote, _ = fit_prior(syn_examples, probes=("none",))

    s = {"probes": PROBES, "shift_semitones": 2, "noise_db": NOISE_DB, "prior_table_synthetic": table,
         "datasets": {}}
    for name, (rows, pr) in data.items():
        print(f"Evaluating {name}")
        ev, _ = evaluate(name, rows, pr["probes"], prior, prior_vote)
        ev["n_disagree_total"] = pr["n_disagree_total"]
        ev["n_rows"] = len(rows)
        ev["rebuild_mismatch"] = pr["mismatch"]
        ev["follow_rates"] = follow_rates(rows, pr["probes"])
        ev["unpitched"] = unpitched_check(rows, pr["probes"], nulls[name], pr)
        # whole dataset: legacy on frames where the lenses agree, the chooser on the rest
        # (MDB: the probed sample stands for all disagreement frames)
        comp = e46.audio_comparator(None)
        agree = [r for r in rows if len(candidates(r["readings"])) < 2]
        a_wrong = np.mean([cents(e46.audio_consensus(r, comp), r["truth"]) > GROSS for r in agree])
        frac = pr["n_disagree_total"] / len(rows)
        ev["whole_dataset_gross"] = {k: float((1 - frac) * a_wrong + frac * ev["gross"][k])
                                     for k in ("legacy", "count", "lowpass_count", "learned")}
        ev["disagree_fraction"] = float(frac)
        if name == "synthetic":
            ev["note"] = "prior is fit on these frames (in-sample); 'learned' is cross-validated"
        s["datasets"][name] = ev
    # ablation: drop one probe from the prior (on real data)
    s["ablation_prior_without"] = {}
    for drop in PROBES[1:]:
        keep = tuple(p for p in PROBES if p != drop)
        f, _ = fit_prior(syn_examples, probes=keep)
        s["ablation_prior_without"][drop] = {
            name: evaluate(name, rows, pr["probes"], f, prior_vote)[0]["gross"]["prior"]
            for name, (rows, pr) in data.items() if name != "synthetic"}
    only = {}
    for p in PROBES:
        f, _ = fit_prior(syn_examples, probes=(p,))
        only[p] = {name: evaluate(name, rows, pr["probes"], f, prior_vote)[0]["gross"]["prior"]
                   for name, (rows, pr) in data.items() if name != "synthetic"}
    s["ablation_prior_only"] = only

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(s, f, indent=2)
    report(s)
    plot(s)
    print(f"\nSaved {OUT_JSON}\nSaved {OUT_PNG}")


def report(s):
    for name, d in s["datasets"].items():
        print(f"\n{name}: {d['n_probed']} probed of {d['n_disagree_total']} disagreement frames "
              f"({d['n_rows']} rows), rebuild mismatches {d['rebuild_mismatch']}")
        g = d["gross"]
        print(f"  q2 answers {100 * d['q2_answer_rate']:.0f}%, gross on its answers "
              f"{100 * d['q2_gross_answered']:.1f}%; on those frames: "
              + ", ".join(f"{k} {100 * v:.1f}%" for k, v in d["on_q2_answered"].items()))
        u = d["unpitched"]
        print(f"  unpitched: {u['n_unpitched']} frames (cached {u['n_unpitched_cached']}, mismatch "
              f"{u['rebuild_mismatch']}), {100 * u['frac_unpitched_with_2plus_candidates']:.0f}% reach the prober, "
              f"AUC {u['auc_best_count_pitched_vs_unpitched']:.2f}")
        for t in (6, 8, 10):
            v = u["thresholds"][t]
            print(f"    abstain below {t} follows: answer {100 * v['answer_pitched']:.0f}%, gross_all "
                  f"{100 * v['gross_all_pitched']:.1f}%, false answers {100 * v['false_answer_unpitched_2plus']:.0f}%")
        print(f"  whole dataset ({100 * d['disagree_fraction']:.0f}% disagree): "
              + ", ".join(f"{k} {100 * v:.1f}%" for k, v in d["whole_dataset_gross"].items()))
        print("  gross error on disagreement frames: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in g.items()))
        for k, v in d["vs_q2"].items():
            print(f"  {k} - q2: {100 * v['diff']:+.1f} pts [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]")
        for k, v in d["vs_legacy"].items():
            print(f"  {k} - legacy: {100 * v['diff']:+.1f} pts [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]")
        for k, v in d["vs_best_lens"].items():
            print(f"  {k} - best lens ({d['best_lens']}): {100 * v['diff']:+.1f} pts "
                  f"[{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]")
        c = d["cost"]
        print(f"  cost: {c['extra_lens_reads_per_frame']:.1f} extra lens reads per frame, "
              f"{c['time_ratio_vs_base_reads']:.1f}x the base reads "
              f"({c['ms_per_frame_base']:.1f} ms -> +{c['ms_per_frame_probe']:.1f} ms)")
    print("\nprior without one probe (real data):", json.dumps(s["ablation_prior_without"]))
    print("prior with one probe only (real data):", json.dumps(s["ablation_prior_only"]))


def plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(s["datasets"])
    methods = ["legacy", "q2", "best_lens", "prior_vote", "prior", "count", "lowpass_count", "learned", "oracle"]
    colors = ["#9aa0a6", "#5f6368", "#c58af9", "#a8c7fa", "#7cacf8", "#1a73e8", "#4285f4", "#0b57d0", "#34a853"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    w = 0.1
    for k, (m, col) in enumerate(zip(methods, colors)):
        vals = [100 * s["datasets"][n]["gross"][m] for n in names]
        ax.bar(np.arange(len(names)) + (k - 4) * w, vals, w, label=m, color=col)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("gross error on disagreement frames (%)")
    ax.set_title("Probing vs voting (q2 counts abstentions as wrong)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    real = [n for n in names if n != "synthetic"]
    labels_ = ["all"] + [f"-{p}" for p in PROBES[1:]] + [f"only {p}" for p in PROBES]
    for k, n in enumerate(real):
        vals = [s["datasets"][n]["gross"]["prior"]] + [s["ablation_prior_without"][p][n] for p in PROBES[1:]] \
            + [s["ablation_prior_only"][p][n] for p in PROBES]
        ax.bar(np.arange(len(labels_)) + (k - 0.5) * 0.38, 100 * np.array(vals), 0.38, label=n,
               color=["#1a73e8", "#f29900"][k])
    ax.set_xticks(np.arange(len(labels_)))
    ax.set_xticklabels(labels_, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("prior chooser gross error (%)")
    ax.set_title("Which probes matter (prior fit on synthetic)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)


if __name__ == "__main__":
    main()
