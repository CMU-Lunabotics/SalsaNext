"""Train-time augmentations for lidar scans.

Each augmentation takes an (N, 4) float32 x, y, z, intensity scan in the lidar frame (plus optional
per-point labels) and returns a new one. Needs numpy only.

A copy of this file lives in the SalsaNext fork as train/common/augmentations.py (the fork runs on
its own on EC2); lidarseg/test_augmentations.py fails if the two copies drift apart.
"""
import os

import numpy as np


def _uniform(rng, lo_hi):
    """A float from [lo, hi], or the value itself for a scalar or an empty range."""
    if np.isscalar(lo_hi):
        return float(lo_hi)
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def _rotvec_to_matrix(w):
    """Axis-angle vector (radians) -> 3x3 rotation matrix (Rodrigues)."""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    kx, ky, kz = w / theta
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


class ScanAugmentation:
    """Applies `apply` to a scan with probability p.

    The random generator is made lazily per process: DataLoader workers are forked from one parent,
    and a generator built there would give every worker the same noise.
    """

    def __init__(self, p=1.0, seed=None):
        self.p = p
        self.seed = seed
        self._rng, self._pid = None, None
        self._home_pid = os.getpid()

    @property
    def rng(self):
        pid = os.getpid()
        if self._pid != pid:
            # a seed repeats exactly in the process that made the augmentation; forked workers mix in
            # their pid so they don't all draw the same noise
            seed = None if self.seed is None else [self.seed, 0 if pid == self._home_pid else pid]
            self._rng, self._pid = np.random.default_rng(seed), pid
        return self._rng

    def __call__(self, points, labels=None):
        if self.p < 1 and self.rng.random() >= self.p:
            return points, labels
        return self.apply(points, labels)

    def apply(self, points, labels):
        raise NotImplementedError


class Compose:
    """Applies augmentations in order."""

    def __init__(self, augs):
        self.augs = list(augs)

    def __call__(self, points, labels=None):
        for aug in self.augs:
            points, labels = aug(points, labels)
        return points, labels


class RegistrationNoise(ScanAugmentation):
    """Smear from imperfect SLAM registration of the scans merged into one frame.

    Real frames merge consecutive scans with their SLAM poses, in the frame of the last scan. Each
    scan's relative pose error is a small rigid transform, so the frame becomes a few slightly offset
    copies of the scene: ground thickens with range (rotation error), edges double (translation).

    The points are split at random into K pseudo-scans (with a non-repetitive pattern every scan covers
    the whole FOV). Each gets a 4x4 transform T_k = [[R_k, t_k], [0, 1]] from a random walk back from
    the reference scan: T_{K-1} = I, T_k = T_{k+1} @ dT_k, where dT_k has translation ~ N(0, trans_std)
    per axis and axis-angle rotation ~ N(0, rot_std_deg) per axis, about the sensor origin.
    Point count, order, intensity and labels are unchanged.

    n_scans, trans_std (m) and rot_std_deg are (min, max) ranges drawn per frame, or fixed values.
    """

    def __init__(self, n_scans=(2, 5), trans_std=(0.005, 0.02), rot_std_deg=(0.05, 0.3), p=1.0, seed=None):
        super().__init__(p, seed)
        self.n_scans, self.trans_std, self.rot_std_deg = n_scans, trans_std, rot_std_deg
        self.last_transforms = None   # (K, 4, 4) from the latest call, for debugging

    def transforms(self):
        rng = self.rng
        lo, hi = (self.n_scans, self.n_scans) if np.isscalar(self.n_scans) else self.n_scans
        k = int(rng.integers(lo, hi + 1))
        t_std = _uniform(rng, self.trans_std)
        r_std = np.radians(_uniform(rng, self.rot_std_deg))
        T = np.tile(np.eye(4), (k, 1, 1))
        for i in range(k - 2, -1, -1):
            dT = np.eye(4)
            dT[:3, :3] = _rotvec_to_matrix(rng.normal(0, r_std, 3))
            dT[:3, 3] = rng.normal(0, t_std, 3)
            T[i] = T[i + 1] @ dT
        return T

    def apply(self, points, labels):
        T = self.transforms()
        self.last_transforms = T
        scan_id = self.rng.integers(len(T), size=len(points))
        xyz = points[:, :3].astype(np.float64)
        R, t = T[scan_id, :3, :3], T[scan_id, :3, 3]
        out = points.copy()
        out[:, :3] = np.einsum('nij,nj->ni', R, xyz) + t
        return out, labels


AUGMENTATIONS = {'registration_noise': RegistrationNoise}


def from_config(cfg):
    """Augmentations from a dict like {'registration_noise': {'p': 0.5, 'n_scans': [2, 5]}}.

    Keys that aren't augmentations here (e.g. SalsaNext's own 'shift', 'drop_points') are skipped, and a
    value of False/None turns one off. Returns a Compose, or None when nothing is on.
    """
    augs = []
    for name, params in (cfg or {}).items():
        if name not in AUGMENTATIONS or params in (False, None):
            continue
        augs.append(AUGMENTATIONS[name](**({} if params is True else params)))
    return Compose(augs) if augs else None


# ---- demo: python -m lidarseg.augmentations --scan 000000.bin --out /tmp/aug ------------------

def _write_ply(path, points, labels):
    colors = np.full((len(points), 3), 160, np.uint8)
    if labels is not None:
        lut = np.random.default_rng(0).integers(40, 255, (int(labels.max()) + 1, 3)).astype(np.uint8)
        colors = lut[labels]
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n'
                f'element vertex {len(points)}\n'
                'property float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        np.savetxt(f, np.column_stack([points[:, :3], colors]), fmt='%.4f %.4f %.4f %d %d %d')


def ground_thickness(points, mask, cell=0.5, ranges=((0, 2), (2, 4), (4, 8))):
    """Residual std (m) of masked points to a plane fit per cell, by range band."""
    xyz = points[mask, :3].astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    cells = np.floor(xyz[:, :2] / cell).astype(np.int64)
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    resid = np.full(len(xyz), np.nan)
    for c in range(inv.max() + 1):
        idx = np.flatnonzero(inv.ravel() == c)
        if len(idx) < 10:
            continue
        A = np.column_stack([xyz[idx, :2], np.ones(len(idx))])
        coef, *_ = np.linalg.lstsq(A, xyz[idx, 2], rcond=None)
        resid[idx] = xyz[idx, 2] - A @ coef
    ok = ~np.isnan(resid)
    return {f'{lo}-{hi} m': float(np.std(resid[ok & (r >= lo) & (r < hi)]))
            if np.any(ok & (r >= lo) & (r < hi)) else float('nan') for lo, hi in ranges}


def _main():
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description='Apply RegistrationNoise to one scan and write before/after .ply')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--scan', type=Path, help='a SemanticKITTI .bin (labels from ../labels/<stem>.label if present)')
    src.add_argument('--kitti', type=Path, help='SemanticKITTI root; uses its first frame')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ground', type=int, default=1, help='semantic id of ground, for the thickness numbers')
    a = ap.parse_args()

    if a.scan:
        points = np.fromfile(a.scan, np.float32).reshape(-1, 4)
        label_file = a.scan.parent.parent / 'labels' / f'{a.scan.stem}.label'
        labels = np.fromfile(label_file, np.uint32) & 0xFFFF if label_file.exists() else None
    else:
        from .readers import iter_kitti
        frame = next(iter_kitti(a.kitti))
        points, labels = frame.points, frame.labels

    aug = RegistrationNoise(seed=a.seed)
    out, out_labels = aug(points, labels)

    assert out.shape == points.shape and out.dtype == points.dtype
    assert out_labels is labels
    assert np.array_equal(out[:, 3], points[:, 3])
    same, _ = RegistrationNoise(trans_std=0, rot_std_deg=0, seed=a.seed)(points, labels)
    assert np.allclose(same, points, atol=1e-5), 'zero noise should give the scan back'

    a.out.mkdir(parents=True, exist_ok=True)
    _write_ply(a.out / 'before.ply', points, labels)
    _write_ply(a.out / 'after.ply', out, labels)
    np.set_printoptions(precision=4, suppress=True)
    print(f'{len(points)} points, {len(aug.last_transforms)} pseudo-scans')
    for i, T in enumerate(aug.last_transforms):
        w = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        print(f'  scan {i}: t = {T[:3, 3]} m, rotation {w:.3f} deg')
    if labels is not None and np.any(labels == a.ground):
        mask = labels == a.ground
        print('ground thickness before:', ground_thickness(points, mask))
        print('ground thickness after: ', ground_thickness(out, mask))
    print('wrote', a.out / 'before.ply', a.out / 'after.ply')


if __name__ == '__main__':
    _main()
