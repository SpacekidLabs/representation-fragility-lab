"""
Experiment 023 — Retune Dynamics Atlas
=======================================
Maps Axis 2: Correction Dynamics.

Axis 1 (Decision Intelligence) is held fixed:
  → Confidence-Gated NoteTracker from Exp 022 selects the target note.

Axis 2 (Correction Dynamics) is swept:
  → Retune speed controls how fast the output pitch moves toward the target.
  → Implemented as a first-order IIR (exponential smoother) on the
     correction amount in semitones:

        correction[n] += alpha * (target[n] - correction[n])

     where alpha is derived from the retune time constant T (ms):

        alpha = 1 - exp(-hop_size / (T * sr / 1000))

     alpha = 1.0  → instant snap (T = 0 ms)
     alpha → 0.0  → never moves  (T → ∞)

Retune speeds tested (ms):
  0, 15, 30, 60, 120, 250, 500

Outputs:
  results/audio/retune_Xms.wav      for each speed X
  listen_test/retune_Xms.wav        (copies for listen test page)
  results/exp023_retune_atlas.png   diagnostic plot
  listen_test/exp023.html           standalone listen test page
"""

import sys
import os
import math
import numpy as np
import scipy.io.wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import librosa

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.representations.acf import compute_acf
from src.experiments.exp017_adaptive_routing import estimate_pitch_acf
from src.experiments.exp022_confidence_gated_tuner import (
    ConfidenceGatedNoteTracker, acf_confidence,
    midi_to_freq, pitch_to_midi, save_wav,
)


# ---------------------------------------------------------------------------
# Retune dynamics engine
# ---------------------------------------------------------------------------

def compute_alpha(retune_ms: float, sr: int, hop: int) -> float:
    """First-order IIR coefficient for a given retune time constant."""
    if retune_ms <= 0.0:
        return 1.0                        # instant snap
    tau_samples = retune_ms * sr / 1000.0
    return 1.0 - math.exp(-hop / tau_samples)


def run_retune(audio: np.ndarray, sr: int, retune_ms: float,
               win_size: int = 4096, hop_size: int = 256) -> dict:
    """
    Run the full pipeline with a fixed retune speed.
    Target note selection: Confidence-Gated NoteTracker (Exp 022).
    Correction dynamics: first-order IIR smoother.
    """
    min_lag = int(sr / 1000)
    max_lag = int(sr / 80)
    alpha = compute_alpha(retune_ms, sr, hop_size)

    tracker = ConfidenceGatedNoteTracker()

    output_audio = np.zeros(len(audio) + win_size)
    window_sum   = np.zeros_like(output_audio)

    # Running correction state (semitones)
    correction_st = 0.0

    log = dict(t=[], detected=[], target_freq=[], target_st=[],
               applied_st=[], confidence=[])

    num_blocks = (len(audio) - win_size) // hop_size + 1

    for n in range(num_blocks):
        start = n * hop_size
        block = audio[start:start + win_size]
        frame = block * np.hanning(win_size)

        acf  = compute_acf(frame)
        conf = acf_confidence(acf, min_lag, max_lag)
        p, _ = estimate_pitch_acf(acf, sr)

        # --- Axis 1: note decision ---
        f_target, _ = tracker.update(p, conf)
        if f_target is None:
            f_target = p   # no commitment → no shift target

        # target semitones relative to detected pitch
        if 80.0 <= p <= 1200.0 and 80.0 <= f_target <= 1200.0:
            target_st = np.clip(12.0 * np.log2(f_target / p), -6.0, 6.0)
        else:
            target_st = 0.0

        # --- Axis 2: correction dynamics (IIR smoother) ---
        correction_st += alpha * (target_st - correction_st)

        # apply
        shifted = librosa.effects.pitch_shift(block, sr=sr, n_steps=correction_st)
        win = np.hanning(win_size)
        output_audio[start:start + win_size] += shifted * win
        window_sum[start:start + win_size]   += win

        log["t"].append((n * hop_size + win_size // 2) / sr)
        log["detected"].append(p)
        log["target_freq"].append(f_target)
        log["target_st"].append(target_st)
        log["applied_st"].append(correction_st)
        log["confidence"].append(conf)

    mask = window_sum > 1e-5
    output_audio[mask] /= window_sum[mask]
    output_audio = output_audio[:len(audio)]
    if np.max(np.abs(output_audio)) > 0:
        output_audio /= np.max(np.abs(output_audio))
        output_audio *= 0.95

    return {"audio": output_audio,
            **{k: np.array(v) for k, v in log.items()},
            "alpha": alpha, "retune_ms": retune_ms}


# ---------------------------------------------------------------------------
# HTML listen-test page generator
# ---------------------------------------------------------------------------

def build_html(speeds: list[float]) -> str:
    note_names = {0: "instant snap", 15: "≈ hard pop", 30: "≈ modern pop",
                  60: "≈ balanced", 120: "≈ natural", 250: "≈ transparent",
                  500: "≈ barely audible"}
    tag_map = {0:   ["robotic", "instant"],
               15:  ["tight", "hard snap"],
               30:  ["modern pop", "fast"],
               60:  ["balanced", "medium"],
               120: ["natural", "slow"],
               250: ["transparent", "very slow"],
               500: ["barely audible", "ultra slow"]}

    colors = ["#888", "#e63946", "#f4a261", "#f5c518",
              "#2ec4b6", "#4cc9f0", "#a8dadc"]
    card_classes = ["card-raw", "card-0", "card-1", "card-2",
                    "card-3", "card-4", "card-5"]

    cards_html = ""
    for i, (ms, col) in enumerate(zip(speeds, colors)):
        fname = f"retune_{int(ms)}ms.wav"
        title = f"{int(ms)} ms" if ms > 0 else "0 ms — Instant Snap"
        subtitle = note_names.get(int(ms), "")
        tags = "".join(f'<span class="tag">{t}</span>' for t in tag_map.get(int(ms), []))
        cards_html += f"""
  <div class="card" id="card-{i}" style="--ca:{col}">
    <div class="card-header">
      <div class="card-label">
        <div class="badge">{i+1}</div>
        <div>
          <div class="card-title">{title}</div>
          <div class="card-subtitle">retune speed · {subtitle}</div>
        </div>
      </div>
      <button class="play-btn" id="btn-{i}" onclick="togglePlay({i})">
        <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
      </button>
    </div>
    <div class="progress-wrap" onclick="seek(event,{i})">
      <div class="progress-bar" id="prog-{i}"></div>
    </div>
    <div class="time-row">
      <span class="time-label" id="cur-{i}">0:00</span>
      <span class="time-label" id="dur-{i}">–:––</span>
    </div>
    <div class="tags">{tags}</div>
    <audio id="audio-{i}" src="{fname}" preload="metadata"></audio>
  </div>"""

    N = len(speeds)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Retune Dynamics Atlas — Exp 023</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{--bg:#0a0a0f;--surface:#111118;--card:#16161f;--border:#1e1e2e;--text:#e2e2f0;--dim:#6b6b88}}
    body{{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;min-height:100vh;
          display:flex;flex-direction:column;align-items:center;padding:48px 24px 80px}}
    header{{text-align:center;margin-bottom:48px}}
    .eyebrow{{font-size:11px;font-weight:600;letter-spacing:.2em;text-transform:uppercase;
              color:#7c6af7;margin-bottom:12px}}
    h1{{font-size:34px;font-weight:700;letter-spacing:-.03em;margin-bottom:12px}}
    header p{{font-size:14px;color:var(--dim);max-width:480px;line-height:1.6}}
    .axis-box{{display:flex;gap:12px;margin-bottom:40px;flex-wrap:wrap;justify-content:center}}
    .axis{{padding:10px 20px;border-radius:12px;border:1px solid var(--border);
           background:var(--surface);font-size:12px;line-height:1.5;max-width:220px;text-align:center}}
    .axis strong{{display:block;font-size:13px;margin-bottom:4px;color:var(--text)}}
    .axis span{{color:var(--dim)}}
    .axis.fixed{{border-color:#7c6af7;}} .axis.sweep{{border-color:#f5a623;}}
    .cards{{display:flex;flex-direction:column;gap:14px;width:100%;max-width:680px}}
    .card{{background:var(--card);border:1px solid var(--border);border-radius:16px;
           padding:24px 28px;position:relative;overflow:hidden;transition:border-color .2s}}
    .card::before{{content:'';position:absolute;top:0;left:0;width:3px;height:100%;
                   background:var(--ca,var(--border));border-radius:3px 0 0 3px}}
    .card.playing{{border-color:var(--ca,var(--border))}}
    .card-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:18px}}
    .card-label{{display:flex;align-items:center;gap:12px}}
    .badge{{width:34px;height:34px;border-radius:9px;background:var(--ca,var(--border));
            display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;
            color:#fff;flex-shrink:0}}
    .card-title{{font-size:15px;font-weight:600;margin-bottom:2px}}
    .card-subtitle{{font-size:11px;color:var(--dim);font-family:'JetBrains Mono',monospace}}
    .play-btn{{width:40px;height:40px;border-radius:50%;background:var(--ca,var(--border));
               border:none;cursor:pointer;display:flex;align-items:center;justify-content:center;
               flex-shrink:0;transition:transform .15s,opacity .15s;opacity:.85;position:relative}}
    .play-btn:hover{{transform:scale(1.08);opacity:1}}
    .play-btn svg{{fill:#fff;width:15px;height:15px}}
    @keyframes pulse{{0%{{transform:scale(1);opacity:.5}}100%{{transform:scale(1.6);opacity:0}}}}
    .play-btn.playing::after{{content:'';position:absolute;width:40px;height:40px;border-radius:50%;
                               background:var(--ca);animation:pulse 1.2s ease-out infinite}}
    .progress-wrap{{height:32px;background:var(--surface);border-radius:8px;overflow:hidden;
                    position:relative;cursor:pointer}}
    .progress-bar{{height:100%;background:var(--ca,var(--border));opacity:.25;width:0%;
                   transition:width .08s linear;border-radius:8px}}
    .progress-wrap:hover .progress-bar{{opacity:.4}}
    .time-row{{display:flex;justify-content:space-between;margin-top:6px}}
    .time-label{{font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--dim)}}
    .tags{{display:flex;gap:6px;flex-wrap:wrap;margin-top:14px}}
    .tag{{font-size:10px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;
          padding:3px 9px;border-radius:99px;border:1px solid var(--border);color:var(--dim)}}
    footer{{margin-top:56px;text-align:center;font-size:12px;color:var(--dim);line-height:1.8}}
  </style>
</head>
<body>
<header>
  <div class="eyebrow">Representation Fragility Lab · Experiment 023</div>
  <h1>Retune Dynamics Atlas</h1>
  <p>Same note target (confidence-gated). Seven correction speeds.<br>Same vocal. Just listen.</p>
</header>
<div class="axis-box">
  <div class="axis fixed">
    <strong>Axis 1 — Fixed</strong>
    <span>Confidence-Gated NoteTracker<br>(Exp 022 decision intelligence)</span>
  </div>
  <div class="axis sweep">
    <strong>Axis 2 — Swept</strong>
    <span>Retune Speed: 0 → 500 ms<br>(correction dynamics)</span>
  </div>
</div>
<div class="cards" id="cards">
{cards_html}
</div>
<footer>
  Experiment 023 · Retune Dynamics Atlas<br>
  <span style="color:#333">Keys 1–{N} to play · Space to pause</span>
</footer>
<script>
  const N={N};
  let cur=-1,timers=Array(N).fill(null);
  const fmt=s=>{{const m=Math.floor(s/60),sec=String(Math.floor(s%60)).padStart(2,'0');return m+':'+sec}};
  for(let i=0;i<N;i++){{
    const a=document.getElementById('audio-'+i);
    a.addEventListener('loadedmetadata',()=>{{document.getElementById('dur-'+i).textContent=fmt(a.duration)}});
    a.addEventListener('ended',stopAll);
  }}
  function stopAll(){{
    for(let i=0;i<N;i++){{
      const a=document.getElementById('audio-'+i);
      a.pause();a.currentTime=0;
      document.getElementById('prog-'+i).style.width='0%';
      document.getElementById('cur-'+i).textContent='0:00';
      document.getElementById('btn-'+i).classList.remove('playing');
      document.getElementById('btn-'+i).innerHTML='<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
      document.getElementById('card-'+i).classList.remove('playing');
      if(timers[i]){{cancelAnimationFrame(timers[i]);timers[i]=null;}}
    }}
    cur=-1;
  }}
  function tick(i){{
    const a=document.getElementById('audio-'+i);
    if(!a.paused&&!a.ended){{
      const p=a.duration?(a.currentTime/a.duration*100):0;
      document.getElementById('prog-'+i).style.width=p+'%';
      document.getElementById('cur-'+i).textContent=fmt(a.currentTime);
      timers[i]=requestAnimationFrame(()=>tick(i));
    }}
  }}
  function togglePlay(i){{
    if(cur===i){{stopAll();return;}}
    stopAll();
    const a=document.getElementById('audio-'+i);
    const btn=document.getElementById('btn-'+i);
    a.play();cur=i;
    btn.classList.add('playing');
    document.getElementById('card-'+i).classList.add('playing');
    btn.innerHTML='<svg viewBox="0 0 24 24"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>';
    timers[i]=requestAnimationFrame(()=>tick(i));
  }}
  function seek(e,i){{
    const a=document.getElementById('audio-'+i);
    if(!a.duration)return;
    const r=e.currentTarget.getBoundingClientRect();
    a.currentTime=(e.clientX-r.left)/r.width*a.duration;
    if(cur!==i)togglePlay(i);
  }}
  document.addEventListener('keydown',e=>{{
    const k=parseInt(e.key);
    if(k>=1&&k<=N)togglePlay(k-1);
    if(e.key===' '){{e.preventDefault();if(cur>=0)togglePlay(cur);}}
  }});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("EXPERIMENT 023 — RETUNE DYNAMICS ATLAS")
    print("=" * 60)

    sr = 22050
    WIN, HOP = 4096, 256

    vocal_path = os.path.join(project_root, "Clean_vocal.wav")
    audio, _ = librosa.load(vocal_path, sr=sr)
    audio /= np.max(np.abs(audio))
    print(f"Loaded: {len(audio)/sr:.2f}s\n")

    SPEEDS = [0, 15, 30, 60, 120, 250, 500]   # ms
    results = {}

    for ms in SPEEDS:
        alpha = compute_alpha(ms, sr, HOP)
        print(f"  Retune {ms:>4d} ms  (alpha={alpha:.4f}) ...")
        res = run_retune(audio, sr, float(ms), win_size=WIN, hop_size=HOP)
        results[ms] = res

        fname = f"retune_{ms}ms.wav"
        save_wav(os.path.join(project_root, "results", "audio", fname), res["audio"], sr)
        save_wav(os.path.join(project_root, "listen_test", fname),       res["audio"], sr)

    print()

    # ------------------------------------------------------------------
    # Diagnostic plot — applied correction trajectory for each speed
    # ------------------------------------------------------------------
    print("Generating diagnostic plot...")

    # Color ramp: red (fast/robotic) → blue (slow/natural)
    cmap = matplotlib.colormaps["cool"]
    speed_colors = [cmap(i / (len(SPEEDS) - 1)) for i in range(len(SPEEDS))]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    plt.subplots_adjust(hspace=0.08)

    ref = results[SPEEDS[0]]   # use 0ms as reference for detected pitch & target

    # Panel 1 — Confidence + target note
    ax0 = axes[0]
    ax0.fill_between(ref["t"], ref["confidence"], alpha=0.2, color="#7c6af7")
    ax0.plot(ref["t"], ref["confidence"], color="#7c6af7", lw=1.5, label="ACF Confidence")
    ax0.axhline(0.60, color="#33cc66", lw=0.8, linestyle="--", alpha=0.6, label="Lock thresh")
    ax0_r = ax0.twinx()
    ax0_r.step(ref["t"], ref["target_freq"], color="#f5a623", lw=1.8, where="mid",
               alpha=0.8, label="Target note (Hz)")
    ax0_r.plot(ref["t"], ref["detected"],    color="#555", lw=1.0, alpha=0.5, label="Detected pitch")
    ax0_r.set_ylabel("Frequency (Hz)", color="#f5a623")
    ax0_r.tick_params(axis='y', colors='#f5a623')
    ax0.set_ylabel("Confidence")
    ax0.set_ylim(-0.05, 1.15)
    ax0.set_title("Axis 1 (Fixed): Confidence-Gated Target Note Selection", fontsize=12, fontweight="bold")
    lines1, labels1 = ax0.get_legend_handles_labels()
    lines2, labels2 = ax0_r.get_legend_handles_labels()
    ax0.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")
    ax0.grid(True, alpha=0.1)

    # Panel 2 — Target semitone shift vs applied (per speed)
    ax1 = axes[1]
    ax1.step(ref["t"], ref["target_st"], color="#f5a623", lw=2.0, where="mid",
             linestyle="--", alpha=0.7, label="Target (instant)", zorder=10)
    for i, ms in enumerate(SPEEDS):
        res = results[ms]
        label = f"{ms} ms" if ms > 0 else "0 ms"
        ax1.plot(res["t"], res["applied_st"], color=speed_colors[i], lw=1.4,
                 alpha=0.85, label=label)
    ax1.axhline(0, color="white", alpha=0.12, lw=0.8)
    ax1.set_ylabel("Applied Correction (semitones)")
    ax1.set_title("Axis 2 (Swept): Correction Trajectory per Retune Speed", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.1)

    # Panel 3 — Correction magnitude (abs) to show "how much processing"
    ax2 = axes[2]
    for i, ms in enumerate(SPEEDS):
        res = results[ms]
        label = f"{ms} ms"
        ax2.fill_between(res["t"], np.abs(res["applied_st"]),
                         alpha=0.15, color=speed_colors[i])
        ax2.plot(res["t"], np.abs(res["applied_st"]),
                 color=speed_colors[i], lw=1.2, alpha=0.9, label=label)
    ax2.set_ylabel("|Correction| (semitones)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_title("Correction Magnitude — 0 ms (top) to 500 ms (bottom)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.1)

    # Colour bar annotation
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colormaps["cool"],
                               norm=plt.Normalize(vmin=0, vmax=500))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.01, aspect=40)
    cbar.set_label("Retune Speed (ms)", rotation=270, labelpad=14, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.text(0.5, 0.002,
             "Fast (red) = instant snap · Slow (blue) = gentle glide · "
             "Same target note, different correction dynamics",
             ha="center", fontsize=9, color="#666")

    out_path = os.path.join(project_root, "results", "exp023_retune_atlas.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")

    # ------------------------------------------------------------------
    # Listen-test HTML
    # ------------------------------------------------------------------
    html = build_html(SPEEDS)
    html_path = os.path.join(project_root, "listen_test", "exp023.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Listen test: {html_path}")

    print("\n" + "=" * 60)
    print("RETUNE DYNAMICS ATLAS COMPLETE")
    print("=" * 60)
    print(f"\n{'Speed (ms)':>12}  {'alpha':>8}  {'Character'}")
    for ms in SPEEDS:
        alpha = results[ms]["alpha"]
        chars = {0:"instant snap", 15:"hard pop", 30:"modern pop",
                 60:"balanced", 120:"natural", 250:"transparent", 500:"barely audible"}
        print(f"{ms:>12}  {alpha:>8.4f}  {chars.get(ms,'')}")
    print()


if __name__ == "__main__":
    run()
