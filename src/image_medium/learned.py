"""
A learned orientation lens: pretrained CNN features + a linear read-out (issue #26, Exp 052)
=========================================================================================
``CnnOrientation`` turns the early layers of an ImageNet-pretrained ResNet-18
into an orientation lens with the core ``Representation`` interface, so it is
compared, voted and patterned like the hand-built lenses in ``medium.py``.

Pinned for reproducibility: torch 2.15.0, torchvision 0.29.2, weights
``ResNet18_Weights.IMAGENET1K_V1`` (``weights_sha256`` reports the hash of the
file actually loaded).

How a patch is read
  1. the gray patch is standardised (mean 0.5, std 0.2: the contrast is removed,
     as CREPE removes the level of a waveform), upsampled 2× (bilinear),
     repeated to 3 channels and normalised with the ImageNet mean and std
  2. the frozen network runs up to ``layer2``; the embedding is the spatial mean
     of the rectified activations after ``conv1`` (64 channels), ``layer1`` (64)
     and ``layer2`` (128): 256 numbers, no orientation built in
  3. a ridge read-out maps the standardised embedding to (cos 2θ, sin 2θ);
     θ = half the angle of that vector, raw contrast = its length
  4. confidence = raw contrast mapped to 0..1 with the rule used for every
     image lens (lo = strongest over 60 white-noise patches, hi = 5th
     percentile over clean patterns)

The read-out is the only trained part. Exp 052 fits it on *clean* synthetic
patterns with seeds and orientations the evaluation never uses, so every
perturbation is as new to it as to the hand-built lenses ("cnn"). A second
read-out ("cnn_aug") is fitted on the same patterns plus noisy and blurred
copies, the usual augmentation; noise and blur are then not new to it. The
fitted heads are saved to ``data/processed/exp052_<name>_head.npz``.
Design choices (layers, standardisation, ridge strength 10) were fixed after a
check of four variants on the synthetic set; all gave 28-35% mean gross error.

torch is imported lazily.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from .medium import Reading, Representation, _scale

PINNED = {"torch": "2.15.0", "torchvision": "0.29.2", "weights": "ResNet18_Weights.IMAGENET1K_V1"}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
_STD = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]


class _Backbone:
    net = None

    @classmethod
    def get(cls):
        if cls.net is None:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18

            torch.set_grad_enabled(False)
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            net.eval()
            cls.net = net
        return cls.net

    @staticmethod
    def weights_sha256() -> str:
        import torch
        from torchvision.models import ResNet18_Weights

        url = ResNet18_Weights.IMAGENET1K_V1.url
        path = f"{torch.hub.get_dir()}/checkpoints/{url.rsplit('/', 1)[-1]}"
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()


def embed(patches: np.ndarray, batch: int = 256, pooling: str = "global") -> np.ndarray:
    """(n, H, W) gray patches in ~[0, 1] → frozen ResNet-18 embedding.

    pooling "global": the spatial mean of each layer's channels, 256 numbers (Exp 052).
    pooling "centre": those 256 plus the mean over the central half of each map
    (512 numbers), so the read-out can tell the structure through the patch centre
    from clutter near the border (Exp 058)."""
    if pooling not in ("global", "centre"):
        raise ValueError(f"unknown pooling {pooling!r}")
    import torch
    import torch.nn.functional as F

    net = _Backbone.get()
    patches = np.asarray(patches, np.float32)
    if patches.ndim == 2:
        patches = patches[None]
    m = patches.mean((1, 2), keepdims=True)
    s = patches.std((1, 2), keepdims=True)
    patches = (0.5 + 0.2 * (patches - m) / np.maximum(s, 1e-6)).astype(np.float32)
    out = []
    for i in range(0, len(patches), batch):
        x = torch.from_numpy(np.repeat(patches[i:i + batch, None], 3, axis=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = (x - torch.from_numpy(_MEAN)) / torch.from_numpy(_STD)
        a0 = net.relu(net.bn1(net.conv1(x)))
        a1 = net.layer1(net.maxpool(a0))
        a2 = net.layer2(a1)
        feats = [a.mean((2, 3)) for a in (a0, a1, a2)]
        if pooling == "centre":
            for a in (a0, a1, a2):
                q = a.shape[2] // 4
                feats.append(a[:, :, q:a.shape[2] - q, q:a.shape[3] - q].mean((2, 3)))
        out.append(torch.cat(feats, 1).numpy())
    return np.concatenate(out).astype(float)


class OrientationHead:
    """Ridge read-out: standardised embedding → (cos 2θ, sin 2θ)."""

    def __init__(self, mu, sd, W, b):
        self.mu, self.sd, self.W, self.b = (np.asarray(v, float) for v in (mu, sd, W, b))

    @classmethod
    def fit(cls, E: np.ndarray, theta_deg: np.ndarray, l2: float = 10.0,
            E_null: np.ndarray | None = None) -> "OrientationHead":
        """Ridge fit. ``E_null``: embeddings of patches with no orientation, fitted to
        (0, 0) so that the read-out's length (its raw contrast) is low on them (Exp 058)."""
        t = np.radians(2 * np.asarray(theta_deg, float))
        Y = np.c_[np.cos(t), np.sin(t)]
        if E_null is not None and len(E_null):
            E = np.r_[E, E_null]
            Y = np.r_[Y, np.zeros((len(E_null), 2))]
        mu, sd = E.mean(0), E.std(0) + 1e-6
        Z = (E - mu) / sd
        b = Y.mean(0)
        W = np.linalg.solve(Z.T @ Z + l2 * len(Z) * np.eye(Z.shape[1]) / 100.0, Z.T @ (Y - b))
        return cls(mu, sd, W, b)

    def predict(self, E: np.ndarray):
        Y = ((E - self.mu) / self.sd) @ self.W + self.b
        theta = (np.degrees(np.arctan2(Y[:, 1], Y[:, 0])) / 2.0) % 180.0
        return theta, np.hypot(Y[:, 0], Y[:, 1])

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd, W=self.W, b=self.b)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mu"], d["sd"], d["W"], d["b"])


class CnnOrientation(Representation):
    """Frozen ImageNet ResNet-18 features with a linear orientation read-out."""

    medium = "image"
    assumptions = ("the patch looks like the clean synthetic patterns the read-out was fitted on",
                   "orientation survives spatial averaging of ImageNet early-layer channels",
                   "the ImageNet input normalisation suits gray patches in [0, 1]")

    def __init__(self, head: OrientationHead, contrast: tuple[float, float] = (0.0, 1.0), name: str = "cnn",
                 pooling: str = "global"):
        self.name = name
        self.head = head
        self.contrast = contrast
        self.pooling = pooling

    def read(self, signal: np.ndarray, context: Mapping[str, Any] | None = None) -> Reading:
        theta, raw, conf = self.read_batch(np.asarray(signal, float)[None])
        return Reading(self.name, float(theta[0]), float(conf[0]), {"raw": float(raw[0])})

    def read_batch(self, patches: np.ndarray):
        theta, raw = self.head.predict(embed(patches, pooling=self.pooling))
        return theta, raw, _scale_many(raw, *self.contrast)


def _scale_many(raw, lo, hi):
    return np.array([_scale(r, lo, hi) for r in np.asarray(raw, float)])
