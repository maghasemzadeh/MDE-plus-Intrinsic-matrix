"""
Multi-domain metric-depth data with exact camera-intrinsics bookkeeping.

Every source dataset is wrapped by a small *reader* that returns the native
RGB image, metric depth (0 = invalid) and the dataset's true pinhole K at
native resolution. All training samples then go through ONE shared
augmentation (`MultiDomainTrainSet`), so indoor, outdoor, real and synthetic
domains are treated identically:

1. zoom-crop: a random window of the native frame (scale 1..zoom_max, 20%
   full frame), K's principal point shifted by the window offset. The window
   is the "virtual camera image": the camera encoder sees (K_window,
   window H, W), exactly what it sees for a full image at test time;
2. horizontal flip, mirroring cx;
3. lower-bound resize of the window to `net` (cubic RGB, nearest depth),
   followed by a random net x net crop. Cropping after the resize does not
   change the focal length in network pixels, which is what metric depth
   depends on;
4. focal jitter: fx, fy and depth scaled by the same log-uniform alpha
   (exact under u = f X / Z, pixels untouched).

The sampler yields (index, aug_seed) pairs drawn from a seeded stream, so two
runs with the same seed (camera-aware model vs K-blind control) see the same
images with the same crops, flips and jitter draws, whatever the worker
scheduling.

Eval readers keep the native frame (no augmentation); `load()` is shared.
"""

import os
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, 'datasets', 'raw_data')

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_K(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class Reader:
    """Base reader: `records` + `load(i) -> (rgb uint8 HxWx3, depth f32 HxW, K)`.

    `max_depth` is the sensor validity cap applied to the *unscaled* GT;
    `eval_range` and `eval_mask()` define the test protocol.
    """
    name = 'base'
    max_depth = 80.0
    eval_range = (1e-3, 80.0)

    def __len__(self):
        return len(self.records)

    def item_id(self, i):
        return str(i)

    def load(self, i):
        raise NotImplementedError

    def eval_mask(self, shape):
        return np.ones(shape, dtype=bool)


# --------------------------------------------------------------------------- NYU
class NYUReader(Reader):
    """NYU Depth V2 labeled set (640x480, Kinect v1).

    split 'train'/'val': the 795 non-eigen images, 80/20 with seed 42 (the
    same leak-free partition as NYUTrainingDataset). split 'test': the
    eigen 654. Training uses the inpainted `depths`; test uses `rawDepths`
    with the eigen crop (verified DA2 protocol).
    """
    name = 'nyu'
    max_depth = 10.0
    eval_range = (1e-3, 10.0)
    K = make_K(518.8579, 519.4696, 325.5824, 253.7362)

    def __init__(self, split='train', depth_field=None, mat_path=None):
        import h5py
        from datasets.nyu import NYUDataset
        mat_path = mat_path or os.path.join(RAW, 'nyu', 'nyu_depth_v2_labeled.mat')
        eigen = sorted(set(NYUDataset.EIGEN_TEST_INDICES))
        if split == 'test':
            idx = eigen
        else:
            non_test = np.array(sorted(set(range(1449)) - set(eigen)))
            np.random.seed(42)
            perm = non_test[np.random.permutation(len(non_test))]
            cut = int(0.8 * len(perm))
            idx = sorted(perm[:cut].tolist() if split == 'train' else perm[cut:].tolist())
        depth_field = depth_field or ('rawDepths' if split == 'test' else 'depths')
        with h5py.File(mat_path, 'r') as h:
            # stored (N, 3, W, H) / (N, W, H) -> (N, H, W, 3) / (N, H, W)
            self.images = np.ascontiguousarray(np.transpose(h['images'][idx], (0, 3, 2, 1)))
            self.depths = np.ascontiguousarray(
                np.transpose(h[depth_field][idx], (0, 2, 1)).astype(np.float32))
        assert self.images.shape[1:] == (480, 640, 3), self.images.shape
        self.records = list(idx)

    def item_id(self, i):
        return f"nyu_{self.records[i]:04d}"

    def load(self, i):
        return self.images[i], self.depths[i], self.K.copy()

    def eval_mask(self, shape):
        m = np.zeros(shape, dtype=bool)
        m[45:471, 41:601] = True  # eigen crop (480x640)
        return m


# ------------------------------------------------------------------------- KITTI
def kitti_K_from_calib(path, cam='02'):
    with open(path) as f:
        for line in f:
            if line.startswith(f'P_rect_{cam}:'):
                P = np.array(line.split(':')[1].split(), dtype=np.float64).reshape(3, 4)
                return make_K(P[0, 0], P[1, 1], P[0, 2], P[1, 2])
    raise ValueError(f"P_rect_{cam} not in {path}")


class KITTIReader(Reader):
    """KITTI with annotated (densified-LiDAR) GT and per-date rectified K.

    train/val: drive-disjoint frames from tools/multidomain/kitti_frames_*.txt
    (every eigen-test drive excluded), RGB as JPEG-95 copies of the raw PNGs.
    test: the DA2 652-image eigen list (raw PNG RGB), no crop, 1e-3..80 m.
    """
    name = 'kitti'
    max_depth = 80.0
    eval_range = (1e-3, 80.0)
    DRIVE_RE = re.compile(r'(\d{4}_\d{2}_\d{2})_drive_\d+_sync')

    def __init__(self, split='train', frames_file=None, jpg_root=None, eval_list=None):
        root = os.path.join(RAW, 'kitti')
        self.jpg_root = jpg_root or os.path.join(root, 'kitti_rgb_jpg')
        self._K = {}
        self.records = []
        if split == 'test':
            eval_list = eval_list or os.path.join(root, 'da2_val.txt')
            for line in open(eval_list).read().splitlines():
                if line.strip():
                    img, gt = line.split()[:2]
                    date = self.DRIVE_RE.search(img).group(1)
                    self.records.append((img, gt, date))
        else:
            frames_file = frames_file or os.path.join(
                PROJECT_ROOT, 'tools', 'multidomain', 'kitti_frames_stride3.txt')
            for line in open(frames_file).read().splitlines():
                if not line.strip():
                    continue
                sp, date, drive, frame, gt = line.split()
                if sp != split:
                    continue
                img = os.path.join(self.jpg_root, date, drive, 'image_02', 'data', f'{frame}.jpg')
                if os.path.exists(img):
                    self.records.append((img, os.path.join(PROJECT_ROOT, gt), date))

    def K_for(self, date):
        if date not in self._K:
            self._K[date] = kitti_K_from_calib(
                os.path.join(self.jpg_root, date, 'calib_cam_to_cam.txt'))
        return self._K[date].copy()

    def item_id(self, i):
        img = self.records[i][0]
        return f"{self.DRIVE_RE.search(img).group(0)}_{os.path.splitext(os.path.basename(img))[0]}"

    def load(self, i):
        img, gt, date = self.records[i]
        rgb = read_rgb(img)
        d = cv2.imread(gt, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise OSError(f"cannot read depth {gt}")
        depth = d.astype(np.float32) / 256.0
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"KITTI size mismatch {img}: {rgb.shape} vs {depth.shape}")
        return rgb, depth, self.K_for(date)


# ------------------------------------------------------------------------ VKITTI
class VKITTIReader(Reader):
    """Virtual KITTI 2 (synthetic, 1242x375). True K from textgt intrinsic.txt
    (constant 725.0087 / 620.5 / 187 across scenes). The .npy intrinsics
    referenced by splits/train.txt are all-zero placeholders and are ignored.
    split 'val' = Scene02 (held out), 'train' = the other four scenes.
    """
    name = 'vkitti'
    max_depth = 80.0
    K = make_K(725.0087, 725.0087, 620.5, 187.0)

    def __init__(self, split='train', max_items=None, seed=0):
        lines = open(os.path.join(RAW, 'vkitti', 'splits', 'train.txt')).read().splitlines()
        recs = []
        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            is_val = '/Scene02/' in parts[0]
            if is_val == (split == 'val'):
                recs.append((parts[0], parts[1]))
        if max_items is not None and len(recs) > max_items:
            rng = np.random.default_rng(seed)
            recs = [recs[j] for j in sorted(rng.choice(len(recs), max_items, replace=False))]
        self.records = recs

    def item_id(self, i):
        return self.records[i][0].split('vkitti_2.0.3_rgb/')[-1]

    def load(self, i):
        img, dp = self.records[i]
        rgb = read_rgb(img)
        d = cv2.imread(dp, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if d is None:
            raise OSError(f"cannot read depth {dp}")
        return rgb, d.astype(np.float32) / 100.0, self.K.copy()


# -------------------------------------------------------------------- Cityscapes
class CityscapesReader(Reader):
    """Cityscapes (2048x1024) with SGM-disparity GT.

    d = (p - 1) / 256 for p > 0 (official encoding), depth = B f / d with the
    camera's baseline 0.209313 m and fx 2262.52 / fy 2265.30 / u0 1096.98 /
    v0 513.14 px. Rows below 80% of the height (ego-vehicle hood) are invalid.
    split 'train' = 18 train cities minus 'zurich'; 'val' = zurich;
    'test' = the official 500-image val split (frankfurt, lindau, munster).
    """
    name = 'cityscapes'
    max_depth = 80.0
    eval_range = (1e-3, 80.0)
    BASELINE = 0.209313
    K = make_K(2262.52, 2265.30, 1096.98, 513.14)
    HOOD_FRAC = 0.8

    def __init__(self, split='train'):
        root = os.path.join(RAW, 'CityScapes')
        src = 'val' if split == 'test' else 'train'
        recs = []
        disp_root = os.path.join(root, 'disparity', src)
        for city in sorted(os.listdir(disp_root)):
            if city.startswith('.') or not os.path.isdir(os.path.join(disp_root, city)):
                continue
            if split == 'train' and city == 'zurich':
                continue
            if split == 'val' and city != 'zurich':
                continue
            for fn in sorted(os.listdir(os.path.join(disp_root, city))):
                if fn.startswith('._') or not fn.endswith('_disparity.png'):
                    continue
                base = fn[:-len('_disparity.png')]
                img = os.path.join(root, 'leftImg8bit', src, city, f'{base}_leftImg8bit.png')
                # (one train image, aachen_000078_000019, is a 0-byte file)
                if os.path.exists(img) and os.path.getsize(img) > 0:
                    recs.append((img, os.path.join(disp_root, city, fn)))
        self.records = recs

    def item_id(self, i):
        return os.path.basename(self.records[i][0]).replace('_leftImg8bit.png', '')

    def load(self, i):
        img, dp = self.records[i]
        rgb = read_rgb(img)
        p = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if p is None:
            raise OSError(f"cannot read disparity {dp}")
        p = p.astype(np.float32)
        disp = np.where(p > 0, (p - 1.0) / 256.0, 0.0)
        depth = np.zeros_like(disp, dtype=np.float32)
        ok = disp > 0
        depth[ok] = self.BASELINE * self.K[0, 0] / disp[ok]
        depth[int(self.HOOD_FRAC * depth.shape[0]):] = 0.0
        return rgb, depth, self.K.copy()


# ------------------------------------------------------------------------- DIODE
class DIODEReader(Reader):
    """DIODE val (1024x768, FARO laser scanner), indoor or outdoor subset.
    K = 886.81 / 927.06 / 512 / 384. Eval caps: indoor 1e-3..10 m, outdoor
    1e-3..80 m (ZoeDepth protocol); the provided validity mask is applied.
    """
    max_depth = 300.0
    K = make_K(886.81, 927.06, 512.0, 384.0)

    def __init__(self, scene_type='indoors'):
        self.name = 'diode_indoor' if scene_type == 'indoors' else 'diode_outdoor'
        self.eval_range = (1e-3, 10.0) if scene_type == 'indoors' else (1e-3, 80.0)
        root = os.path.join(RAW, 'diode', 'val', scene_type)
        recs = []
        for dirpath, _, files in os.walk(root):
            for fn in sorted(files):
                if fn.endswith('.png'):
                    stem = os.path.join(dirpath, fn[:-4])
                    recs.append((stem + '.png', stem + '_depth.npy', stem + '_depth_mask.npy'))
        self.records = sorted(recs)

    def item_id(self, i):
        return os.path.basename(self.records[i][0])[:-4]

    def load(self, i):
        img, dp, mp = self.records[i]
        rgb = read_rgb(img)
        depth = np.load(dp).squeeze().astype(np.float32)
        mask = np.load(mp).squeeze() > 0
        depth[~mask] = 0.0
        return rgb, depth, self.K.copy()


# -------------------------------------------------------------------- Middlebury
class MiddleburyReader(Reader):
    """Middlebury 2014 '-perfect' scenes, im0 + disp0, at the standard
    MiddEval3 'H' (half) resolution: image area-downsampled 2x, depth
    subsampled 2x (values unchanged), K halved. The full ~2880x1988 frames lie
    far outside every training image size.
    depth = baseline * f / (d + doffs) from each scene's calib.txt (mm -> m).
    """
    name = 'middlebury'
    max_depth = 20.0
    eval_range = (1e-3, 20.0)

    def __init__(self):
        root = os.path.join(RAW, 'middlebury')
        self.records = [os.path.join(root, s) for s in sorted(os.listdir(root))
                        if s.endswith('-perfect') and os.path.exists(os.path.join(root, s, 'disp0.pfm'))]

    def item_id(self, i):
        return os.path.basename(self.records[i])

    def load(self, i):
        from datasets.training_datasets import _parse_middlebury_calib, _read_pfm
        scene = self.records[i]
        f0, _, cx0, cy0, _, _, doffs, baseline = _parse_middlebury_calib(
            os.path.join(scene, 'calib.txt'))
        rgb = read_rgb(os.path.join(scene, 'im0.png'))
        disp = _read_pfm(os.path.join(scene, 'disp0.pfm')).astype(np.float32)
        depth = np.zeros_like(disp, dtype=np.float32)
        ok = np.isfinite(disp) & (disp > 0)
        depth[ok] = (baseline / 1000.0) * f0 / (disp[ok] + doffs)
        h2, w2 = rgb.shape[0] // 2, rgb.shape[1] // 2
        rgb = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_AREA)
        depth = np.ascontiguousarray(depth[:2 * h2:2, :2 * w2:2])
        return rgb, depth, make_K(f0 / 2, f0 / 2, (cx0 + 0.5) / 2 - 0.5, (cy0 + 0.5) / 2 - 0.5)


# ----------------------------------------------------------------- training set
def _focal_jitter_alpha(rng, m):
    if not m or m <= 1.0:
        return 1.0
    return float(np.exp(rng.uniform(-np.log(m), np.log(m))))


class MultiDomainTrainSet(Dataset):
    """Concatenation of readers with the shared exact-geometry augmentation."""

    def __init__(self, readers, net=518, zoom_max=1.0, p_full=0.2, focal_jitter=0.0,
                 flip=True, aspect_max=0.0, res_jitter=0.0, jpeg_aug=False):
        self.readers = readers
        self.offsets = np.cumsum([0] + [len(r) for r in readers])
        self.net = net
        self.zoom_max = zoom_max
        self.p_full = p_full
        self.focal_jitter = focal_jitter
        self.flip = flip
        # Shortcut breakers for the camera encoder (both exact): random window
        # aspect in [1, aspect_max] so aspect ratio stops identifying the
        # dataset, and a random virtual resolution r in [1/res_jitter,
        # res_jitter] (K and image size scaled together, pixels untouched) so
        # the encoder's absolute image-area feature carries no information.
        self.aspect_max = aspect_max
        self.res_jitter = res_jitter
        # KITTI training frames are JPEG copies (and VKITTI ships JPEG) while
        # NYU/Cityscapes are lossless: randomly re-compress every domain so
        # compression artefacts cannot act as a dataset identifier
        self.jpeg_aug = jpeg_aug

    def __len__(self):
        return int(self.offsets[-1])

    def locate(self, gidx):
        d = int(np.searchsorted(self.offsets, gidx, side='right') - 1)
        return d, int(gidx - self.offsets[d])

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            gidx, seed = int(index[0]), int(index[1])
        else:  # RobustDataset fallback / plain sampler
            gidx, seed = int(index), int(np.random.randint(2 ** 62))
        rng = np.random.default_rng(seed)
        d, li = self.locate(gidx)
        reader = self.readers[d]
        for attempt in range(10):
            try:
                rgb, depth, K = reader.load(li)
                break
            except (OSError, ValueError):
                # deterministic same-domain fallback keeps paired runs paired
                li = int(rng.integers(len(reader)))
        else:
            raise OSError(f"{reader.name}: 10 consecutive unreadable samples")
        depth = depth.astype(np.float32, copy=True)
        valid = (depth > 0) & (depth <= reader.max_depth) & np.isfinite(depth)
        K = K.copy()
        H, W = depth.shape

        # 1. zoom-crop window (aspect preserved) = virtual camera image
        s = 1.0 if (self.zoom_max <= 1.0 or rng.random() < self.p_full) \
            else rng.uniform(1.0, self.zoom_max)
        wh, ww = int(round(H / s)), int(round(W / s))
        if self.aspect_max > 1.0:
            a = float(np.exp(rng.uniform(0.0, np.log(self.aspect_max))))
            ww = int(min(W, max(wh, round(a * wh))))
        y0 = int(rng.integers(0, H - wh + 1))
        x0 = int(rng.integers(0, W - ww + 1))
        rgb = rgb[y0:y0 + wh, x0:x0 + ww]
        depth = depth[y0:y0 + wh, x0:x0 + ww]
        valid = valid[y0:y0 + wh, x0:x0 + ww]
        K[0, 2] -= x0
        K[1, 2] -= y0

        if self.jpeg_aug and rng.random() < 0.5:
            q = int(rng.integers(70, 96))
            _, buf = cv2.imencode('.jpg', np.ascontiguousarray(rgb[..., ::-1]),
                                  [cv2.IMWRITE_JPEG_QUALITY, q])
            rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]

        # 2. horizontal flip
        if self.flip and rng.random() < 0.5:
            rgb, depth, valid = rgb[:, ::-1], depth[:, ::-1], valid[:, ::-1]
            K[0, 2] = (ww - 1) - K[0, 2]

        # 3. lower-bound resize to `net`, then random net x net crop
        r = self.net / min(wh, ww)
        nh, nw = max(self.net, int(round(wh * r))), max(self.net, int(round(ww * r)))
        rgb = cv2.resize(np.ascontiguousarray(rgb), (nw, nh), interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(np.ascontiguousarray(depth), (nw, nh), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(np.ascontiguousarray(valid).astype(np.uint8), (nw, nh),
                           interpolation=cv2.INTER_NEAREST) > 0
        cy0 = int(rng.integers(0, nh - self.net + 1))
        cx0 = int(rng.integers(0, nw - self.net + 1))
        sl = (slice(cy0, cy0 + self.net), slice(cx0, cx0 + self.net))
        rgb, depth, valid = rgb[sl], depth[sl], valid[sl]

        # 4. focal jitter (exact: same image is consistent with (a f, a Z))
        alpha = _focal_jitter_alpha(rng, self.focal_jitter)
        K[0, 0] *= alpha
        K[1, 1] *= alpha
        depth = depth * alpha

        # 5. virtual-resolution jitter (encoder input only)
        size = np.array([wh, ww], dtype=np.float64)
        if self.res_jitter > 1.0:
            res = float(np.exp(rng.uniform(-np.log(self.res_jitter), np.log(self.res_jitter))))
            K[:2] *= res
            size *= res

        img = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return {
            'image': torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
            'depth': torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32)),
            'valid_mask': torch.from_numpy(np.ascontiguousarray(valid)),
            'intrinsics': torch.from_numpy(K.astype(np.float32)),
            'original_size': torch.from_numpy(size.astype(np.float32)),
            'domain': torch.tensor(d, dtype=torch.long),
            'jitter_alpha': torch.tensor(alpha, dtype=torch.float32),
        }


class DomainBalancedSampler(Sampler):
    """Each epoch draws `per_domain[d]` samples from domain d (cycling through
    fresh permutations, so small domains repeat with new augmentation draws),
    shuffles them together and yields (global_index, aug_seed) pairs from a
    stream seeded by (seed, epoch) -- identical for paired runs."""

    def __init__(self, dataset, per_domain, seed=0):
        self.offsets = dataset.offsets
        self.sizes = [int(self.offsets[i + 1] - self.offsets[i]) for i in range(len(per_domain))]
        self.per_domain = [int(n) for n in per_domain]
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return sum(self.per_domain)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        items = []
        for d, (n, size) in enumerate(zip(self.per_domain, self.sizes)):
            reps = -(-n // size)
            local = np.concatenate([rng.permutation(size) for _ in range(reps)])[:n]
            items.append(local + int(self.offsets[d]))
        items = np.concatenate(items)
        items = items[rng.permutation(len(items))]
        seeds = rng.integers(0, 2 ** 62, size=len(items))
        return iter([(int(i), int(s)) for i, s in zip(items, seeds)])


class MultiDomainValSet(Dataset):
    """Native-frame validation samples for in-training monitoring: image at
    the inference resolution (lower-bound 518, multiple of 14), GT at native
    resolution, native K and size."""

    def __init__(self, readers, per_domain=50, net=518, seed=0):
        self.readers = readers
        self.net = net
        rng = np.random.default_rng(seed)
        self.items = []
        for d, r in enumerate(readers):
            n = min(per_domain, len(r))
            for li in sorted(rng.choice(len(r), n, replace=False)):
                self.items.append((d, int(li)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, li = self.items[i]
        reader = self.readers[d]
        rgb, depth, K = reader.load(li)
        valid = (depth > 0) & (depth <= reader.max_depth) & np.isfinite(depth)
        H, W = depth.shape
        r = self.net / min(H, W)
        nh = max(self.net, int(round(H * r / 14)) * 14)
        nw = max(self.net, int(round(W * r / 14)) * 14)
        img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return {
            'image': torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
            'depth': torch.from_numpy(depth.astype(np.float32)),
            'valid_mask': torch.from_numpy(valid),
            'intrinsics': torch.from_numpy(K.astype(np.float32)),
            'original_size': torch.tensor([H, W], dtype=torch.long),
            'domain': torch.tensor(d, dtype=torch.long),
        }


TRAIN_READERS = {
    'nyu': lambda split: NYUReader(split),
    'kitti': lambda split: KITTIReader(split),
    'vkitti': lambda split: VKITTIReader(split, max_items=None if split == 'train' else 400),
    'cityscapes': lambda split: CityscapesReader(split),
}


def build_multidomain(spec, net=518, zoom_max=1.0, focal_jitter=0.0, seed=0, val_per_domain=50,
                      aspect_max=0.0, res_jitter=0.0, jpeg_aug=False):
    """spec: 'nyu:4000,kitti:4000,...' (domain:samples-per-epoch).
    Returns (trainset, sampler, valset, domain_names)."""
    names, counts = [], []
    for part in spec.split(','):
        name, n = part.split(':')
        names.append(name.strip())
        counts.append(int(n))
    train_readers = [TRAIN_READERS[n]('train') for n in names]
    val_readers = [TRAIN_READERS[n]('val') for n in names]
    trainset = MultiDomainTrainSet(train_readers, net=net, zoom_max=zoom_max,
                                   focal_jitter=focal_jitter, aspect_max=aspect_max,
                                   res_jitter=res_jitter, jpeg_aug=jpeg_aug)
    sampler = DomainBalancedSampler(trainset, counts, seed=seed)
    valset = MultiDomainValSet(val_readers, per_domain=val_per_domain, net=net, seed=seed)
    return trainset, sampler, valset, names


EVAL_READERS = {
    'nyu': lambda: NYUReader('test'),
    'kitti': lambda: KITTIReader('test'),
    'cityscapes': lambda: CityscapesReader('test'),
    'diode_indoor': lambda: DIODEReader('indoors'),
    'diode_outdoor': lambda: DIODEReader('outdoor'),
    'middlebury': lambda: MiddleburyReader(),
}


# --------------------------------------------------------------------- SUN RGB-D
class SUNRGBDReader(Reader):
    """SUN RGB-D raw depth, one sensor group per reader (zero-shot indoor):
    kv2 (Kinect v2, 730x530, f 529.5), realsense (681x531, f ~694),
    xtion (591x441, f ~570), kv1 (Kinect v1 B3DO, 561x427, f 520.5).
    Random 200-frame subsets fetched by tools/multidomain/fetch_sunrgbd.py;
    kv1/NYUdata is excluded there (it is the NYU labeled set). K is the
    per-frame intrinsics.txt (3x3 row-major). Depth PNGs store the toolbox's
    bit-rotated millimetres: d = ((p >> 3) | (p << 13)) & 0xFFFF, / 1000.
    Eval caps 1e-3..10 m.
    """
    max_depth = 10.0
    eval_range = (1e-3, 10.0)

    def __init__(self, sensor):
        self.name = f'sunrgbd_{sensor}'
        root = os.path.join(RAW, 'sunrgbd_subset', sensor)
        self.records = sorted(os.path.join(root, f) for f in os.listdir(root)
                              if os.path.exists(os.path.join(root, f, 'intrinsics.txt')))

    def item_id(self, i):
        return os.path.basename(self.records[i])

    def load(self, i):
        d = self.records[i]
        rgb = read_rgb(os.path.join(d, 'image.jpg'))
        p = cv2.imread(os.path.join(d, 'depth.png'), cv2.IMREAD_UNCHANGED)
        if p is None:
            raise OSError(f"cannot read depth {d}")
        p = p.astype(np.uint32)
        depth = (((p >> 3) | (p << 13)) & 0xFFFF).astype(np.float32) / 1000.0
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"SUN RGB-D size mismatch {d}: {rgb.shape} vs {depth.shape}")
        K = np.loadtxt(os.path.join(d, 'intrinsics.txt')).reshape(3, 3)
        return rgb, depth, make_K(K[0, 0], K[1, 1], K[0, 2], K[1, 2])


EVAL_READERS.update({
    f'sunrgbd_{s}': (lambda s=s: SUNRGBDReader(s)) for s in ('kv2', 'realsense', 'xtion', 'kv1')
})
