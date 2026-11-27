# Data

Real-world datasets with ground truth, used by Exp 045 (issue #7). They are **not**
committed: `data/raw/` (downloads) and `data/processed/` (cached rows) are in
`.gitignore`. Fetch them with the commands below; each goes into its own folder.

| Dataset | What | Ground truth | Size | License / source |
|---|---|---|---|---|
| vocadito | 40 solo singing excerpts, 7 languages (13.6 min) | frame-level f0, annotated by musicians | 58 MB | CC BY 4.0, [Zenodo 5578807](https://zenodo.org/records/5578807) (Bittner et al. 2021) |
| MDB-stem-synth | 230 MedleyDB stems (voice and instruments), resynthesised | exact frame-level f0 (the resynthesis follows the annotation) | 1.8 GB | CC BY-NC 4.0, [Zenodo 1481172](https://zenodo.org/records/1481172) (Salamon et al. 2017) |
| MIT-BIH Arrhythmia | 48 two-lead ECG records, 30 min each at 360 Hz (Exp 053) | every beat annotated by two cardiologists | 77 MB | ODC-By 1.0, [PhysioNet mitdb](https://physionet.org/content/mitdb/1.0.0/) (Moody & Mark 2001) |
| MIT-BIH Noise Stress Test | electrode-motion, muscle and baseline-wander noise recorded without ECG (Exp 053) | — (used as real perturbations and as rhythm-free input) | 71 MB | ODC-By 1.0, [PhysioNet nstdb](https://physionet.org/content/nstdb/1.0.0/) (Moody et al. 1984) |
| BSDS500 | 500 natural photos (481×321) | ~5 human boundary maps per image | 71 MB | [Berkeley BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) (Arbeláez et al. 2011) |

```bash
mkdir -p data/raw/vocadito data/raw/mdb_stem_synth data/raw/bsds500
curl -L -o data/raw/vocadito/vocadito.zip https://zenodo.org/api/records/5578807/files/vocadito.zip/content
curl -L -o data/raw/mdb_stem_synth/MDB-stem-synth.tar.gz https://zenodo.org/api/records/1481172/files/MDB-stem-synth.tar.gz/content
curl -L -o data/raw/bsds500/BSR_bsds500.tgz https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
unzip -q data/raw/vocadito/vocadito.zip -d data/raw/vocadito
tar xzf data/raw/mdb_stem_synth/MDB-stem-synth.tar.gz -C data/raw/mdb_stem_synth
tar xzf data/raw/bsds500/BSR_bsds500.tgz -C data/raw/bsds500
mkdir -p data/raw/mitdb data/raw/nstdb
curl -L -o data/raw/mitdb/mitdb.zip https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
curl -L -o data/raw/nstdb/nstdb.zip https://physionet.org/static/published-projects/nstdb/mit-bih-noise-stress-test-database-1.0.0.zip
unzip -q data/raw/mitdb/mitdb.zip -d data/raw/mitdb
unzip -q data/raw/nstdb/nstdb.zip -d data/raw/nstdb
```

Expected layout:

```
data/raw/vocadito/Audio/vocadito_*.wav, Annotations/F0/vocadito_*_f0.csv
data/raw/mdb_stem_synth/MDB-stem-synth/audio_stems/*.wav, annotation_stems/*.csv
data/raw/bsds500/BSR/BSDS500/data/images/{train,val,test}/*.jpg, groundTruth/{train,val,test}/*.mat
data/raw/mitdb/mit-bih-arrhythmia-database-1.0.0/{100..234}.{hea,dat,atr}
data/raw/nstdb/mit-bih-noise-stress-test-database-1.0.0/{em,ma,bw}.{hea,dat}
```

Each PhysioNet archive ships a `SHA256SUMS.txt`; Exp 053 reads the WFDB files
(format 212 signals, MIT annotations) with numpy, no extra package.

Exp 045 skips any dataset that is missing and caches its extracted rows in
`data/processed/` (`--refresh` recomputes them).
