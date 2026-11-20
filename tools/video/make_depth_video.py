#!/usr/bin/env python3
"""Render a driving-video demo of a train.py depth checkpoint.

Top panel: the camera frame. Bottom panel: the metric depth the model predicts
for that frame (one frame at a time, no temporal smoothing), drawn with one
fixed colour scale for the whole video. A footer shows the resolution, the
measured inference speed, the hardware and, where KITTI annotated depth exists
for the frame, the error against LiDAR. The video ends with a summary card.

Frames come from KITTI raw drives laid out like the raw dataset:
    <frames-root>/<date>/<drive>/image_02/data/<frame>.png
    <calib-root>/<date>/calib_cam_to_cam.txt          (rectified K per date)
    <gt-root>/<drive>/proj_depth/groundtruth/image_02/<frame>.png   (optional)

Timing: every stage is bracketed by torch.cuda.synchronize().
    jpeg  optional JPEG re-encode of the frame (--jpeg-quality)
    pre   resize to the network input, normalise, upload
    net   forward pass
    post  upsample to the camera resolution and download
"end-to-end" = jpeg + pre + net + post, i.e. decoded camera frame in, depth map
out. Reading the file, drawing and video encoding are not counted.

--slow-pre uses model.image2tensor unchanged (float64 cubic resize on the CPU,
as in the evaluation scripts). The default does the same cubic resize on the
uint8 frame and normalises on the GPU: same network input up to rounding,
about 15 ms less per KITTI frame.

--no-video only measures (speed and error), e.g. to compare --amp settings.

--layout plain writes the two panels only (no header, footer or summary card),
centre-cropped to --plain-aspect, for embedding as a silent loop, e.g. in the
narrow column of the defense deck's title slide. The model still sees the whole
frame; only the drawing is cropped.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools', 'multidomain'))

from eval_multidomain import depth_metrics, load_model  # noqa: E402
from datasets.multidomain import kitti_K_from_calib  # noqa: E402

FONT_DIR = '/usr/share/fonts/truetype/dejavu'
BG, PANEL, INK, MUTED, ACCENT = (13, 16, 22), (24, 29, 38), (232, 236, 242), (139, 148, 163), (94, 214, 190)
AMP = {'none': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}
CAMERA_HZ = 10.0  # KITTI capture rate


def font(size, bold=False):
    return ImageFont.truetype(os.path.join(FONT_DIR, 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'),
                              int(round(size)))


def even(x):
    return int(round(x / 2.0)) * 2


def spectral_lut():
    """256 RGB colours, index 0 = near (warm), 255 = far (cool)."""
    import matplotlib
    cmap = matplotlib.colormaps['Spectral']
    return (cmap(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)


def colorize(depth, lo, hi, lut):
    x = (np.log(np.clip(depth, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return lut[np.round(x * 255).astype(np.uint8)]


_NET_SIZE, _NORM = {}, {}


def fast_tensor(model, bgr, input_size, device):
    """Same network input as model.image2tensor up to rounding: cubic resize of
    the uint8 frame on the CPU, colour order and normalisation on the GPU."""
    h, w = bgr.shape[:2]
    key = (h, w, input_size)
    if key not in _NET_SIZE:  # target size from the reference transform
        _NET_SIZE[key] = tuple(model.image2tensor(bgr, input_size)[0].shape[-2:])
        _NORM['mean'] = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255.0
        _NORM['std'] = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255.0
    nh, nw = _NET_SIZE[key]
    small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = torch.from_numpy(small).to(device).permute(2, 0, 1)[None].flip(1).float()
    return (x - _NORM['mean']) / _NORM['std'], (h, w)


@torch.no_grad()
def infer(model, bgr, K, input_size, amp_dtype, device, slow_pre=False, jpeg_quality=0):
    """model.infer_image with the stages timed. Returns depth (H, W) in metres,
    the network input size and (jpeg, pre, net, post) seconds."""
    tj = time.perf_counter()
    if jpeg_quality:
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if slow_pre:
        image, (h, w) = model.image2tensor(bgr, input_size)
    else:
        image, (h, w) = fast_tensor(model, bgr, input_size, device)
    Kt = torch.from_numpy(K).float().unsqueeze(0).to(device) if K is not None else None
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
        depth = model.forward(image, intrinsics=Kt, image_size=(h, w))
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    depth = F.interpolate(depth[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0]
    out = depth.cpu().numpy()
    t3 = time.perf_counter()
    return out, tuple(image.shape[-2:]), (t0 - tj, t1 - t0, t2 - t1, t3 - t2)


class Segment:
    def __init__(self, spec, args):
        parts = spec.split(':')
        self.drive = parts[0]
        self.date = self.drive[:10]
        d = os.path.join(args.frames_root, self.date, self.drive, 'image_02', 'data')
        frames = sorted(f for f in os.listdir(d) if f.endswith('.png'))
        start = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        end = int(parts[2]) if len(parts) > 2 and parts[2] else len(frames)
        self.frames = [os.path.join(d, f) for f in frames[start:end:args.stride]]
        self.gt_dir = os.path.join(args.gt_root, self.drive, 'proj_depth', 'groundtruth', 'image_02')
        self.K = kitti_K_from_calib(os.path.join(args.calib_root, self.date, 'calib_cam_to_cam.txt'))
        self.label = f"KITTI raw · {self.date} · drive {self.drive.split('_')[4]}"

    def gt(self, frame_path):
        p = os.path.join(self.gt_dir, os.path.basename(frame_path))
        if not os.path.exists(p):
            return None
        return cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0


class Canvas:
    """Fixed layout: header, camera panel, depth panel, footer."""

    def __init__(self, scale, lo, hi, lut, static):
        s = self.s = scale
        self.lo, self.hi, self.lut, self.static = lo, hi, lut, static
        self.W, self.PH = even(1242 * s), even(375 * s)
        self.header, self.gap, self.footer = int(62 * s), int(4 * s), int(112 * s)
        self.y_rgb = self.header
        self.y_depth = self.header + self.PH + self.gap
        self.y_foot = self.y_depth + self.PH
        self.H = even(self.y_foot + self.footer)
        self.f_title, self.f_sub = font(21 * s, True), font(12.5 * s)
        self.f_tag, self.f_probe = font(11 * s, True), font(13 * s, True)
        self.f_key, self.f_big, self.f_small = font(9.5 * s, True), font(21 * s, True), font(10.5 * s)
        self.f_tick = font(9.5 * s)
        self.probes = [(0.22, 0.62), (0.50, 0.60), (0.78, 0.62)]

    def panel(self, rgb):
        return cv2.resize(rgb, (self.W, self.PH), interpolation=cv2.INTER_LANCZOS4)

    def _tag(self, d, x, y, text):
        s = self.s
        w = d.textlength(text, font=self.f_tag)
        d.rounded_rectangle([x, y, x + w + 16 * s, y + 22 * s], radius=5 * s, fill=(0, 0, 0, 150))
        d.text((x + 8 * s, y + 4 * s), text, font=self.f_tag, fill=INK)

    def _tile(self, d, x, w, key, big, small1, small2=''):
        s, y = self.s, self.y_foot + 12 * self.s
        d.rounded_rectangle([x, y, x + w, self.H - 12 * s], radius=7 * s, fill=PANEL)
        d.text((x + 12 * s, y + 9 * s), key, font=self.f_key, fill=ACCENT)
        d.text((x + 12 * s, y + 24 * s), big, font=self.f_big, fill=INK)
        d.text((x + 12 * s, y + 54 * s), small1, font=self.f_small, fill=MUTED)
        d.text((x + 12 * s, y + 69 * s), small2, font=self.f_small, fill=MUTED)

    def _colorbar(self, img, d, x, w):
        s, y = self.s, self.y_foot + 12 * self.s
        d.rounded_rectangle([x, y, x + w, self.H - 12 * s], radius=7 * s, fill=PANEL)
        d.text((x + 12 * s, y + 9 * s), 'DEPTH COLOUR SCALE (metres, log)', font=self.f_key, fill=ACCENT)
        bx, bw, by, bh = int(x + 14 * s), int(w - 28 * s), int(y + 32 * s), int(20 * s)
        bar = self.lut[np.linspace(0, 255, bw).astype(np.uint8)][None].repeat(bh, 0)
        img.paste(Image.fromarray(bar), (bx, by))
        for m in (3, 5, 10, 20, 40, 80):
            if not self.lo <= m <= self.hi:
                continue
            fx = bx + (np.log(m) - np.log(self.lo)) / (np.log(self.hi) - np.log(self.lo)) * (bw - 1)
            d.line([fx, by + bh, fx, by + bh + 5 * s], fill=MUTED, width=max(1, int(s)))
            t = str(m)
            tw = d.textlength(t, font=self.f_tick)
            d.text((min(max(fx - tw / 2, bx), bx + bw - tw), by + bh + 7 * s), t, font=self.f_tick, fill=INK)
        d.text((bx, by + bh + 22 * s), 'near', font=self.f_tick, fill=MUTED)
        far = f'far (≥ {self.hi:g} m)'
        d.text((bx + bw - d.textlength(far, font=self.f_tick), by + bh + 22 * s), far, font=self.f_tick, fill=MUTED)

    def frame(self, rgb, depth, seg, idx, n, live):
        s = self.s
        img = Image.new('RGB', (self.W, self.H), BG)
        img.paste(Image.fromarray(self.panel(rgb)), (0, self.y_rgb))
        img.paste(Image.fromarray(self.panel(colorize(depth, self.lo, self.hi, self.lut))), (0, self.y_depth))
        d = ImageDraw.Draw(img, 'RGBA')

        d.text((16 * s, 9 * s), 'Metric depth from a single driving camera', font=self.f_title, fill=INK)
        d.text((16 * s, 38 * s), self.static['model_line'], font=self.f_sub, fill=MUTED)
        right = f"frame {idx + 1} / {n}   ·   {idx / CAMERA_HZ:5.1f} s"
        d.text((self.W - 16 * s - d.textlength(seg.label, font=self.f_sub), 13 * s), seg.label,
               font=self.f_sub, fill=INK)
        d.text((self.W - 16 * s - d.textlength(right, font=self.f_sub), 35 * s), right, font=self.f_sub, fill=MUTED)

        h, w = depth.shape
        self._tag(d, 12 * s, self.y_rgb + 10 * s, f'CAMERA FRAME   {w} × {h} px')
        self._tag(d, 12 * s, self.y_depth + 10 * s, 'PREDICTED DEPTH   one frame at a time, no temporal smoothing')

        for px, py in self.probes:
            yy, xx = int(py * h), int(px * w)
            dist = float(np.median(depth[yy - 2:yy + 3, xx - 2:xx + 3]))
            cx, r = px * self.W, 5 * s
            for y0 in (self.y_rgb, self.y_depth):
                cy = y0 + py * self.PH
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, 255), width=int(2 * s))
                d.ellipse([cx - 1.5 * s, cy - 1.5 * s, cx + 1.5 * s, cy + 1.5 * s], fill=(255, 255, 255, 255))
            t = f'{dist:.1f} m'
            tw = d.textlength(t, font=self.f_probe)
            cy = self.y_depth + py * self.PH
            d.rounded_rectangle([cx - tw / 2 - 8 * s, cy - 34 * s, cx + tw / 2 + 8 * s, cy - 11 * s],
                                radius=5 * s, fill=(0, 0, 0, 175))
            d.text((cx - tw / 2, cy - 31 * s), t, font=self.f_probe, fill=(255, 255, 255))

        m, x, g = 12 * s, 12 * s, 10 * s
        widths = [0.17, 0.235, 0.20, 0.19]
        tiles = [
            ('RESOLUTION', f'{w} × {h}', f"network input {live['net_w']} × {live['net_h']}",
             f"{live['net_w'] // 14 * (live['net_h'] // 14):,} image patches"),
            ('SPEED (measured, end-to-end)', f"{live['fps']:.1f} FPS",
             f"{live['ms']:.0f} ms per frame · network alone {live['net_ms']:.0f} ms",
             f"{live['fps'] / CAMERA_HZ:.1f}× the camera rate ({CAMERA_HZ:g} Hz)"),
            ('ERROR vs LiDAR (this frame)',
             f"AbsRel {live['abs_rel']:.3f}" if live['abs_rel'] is not None else 'AbsRel  —',
             (f"δ1 {live['d1']:.3f} · RMSE {live['rmse']:.2f} m" if live['abs_rel'] is not None
              else 'no annotated depth for this frame'),
             f"drive so far: AbsRel {live['run_abs_rel']:.3f}" if live['run_abs_rel'] is not None else ''),
            ('HARDWARE', self.static['gpu_short'], f"{self.static['precision'].split()[0]} · {self.static['params_m']:.0f} M parameters",
             f"camera K given: fx {seg.K[0, 0]:.0f} px"),
        ]
        for (key, big, s1, s2), fw in zip(tiles, widths):
            tw = fw * (self.W - 2 * m) - g
            self._tile(d, x, tw, key, big, s1, s2)
            x += tw + g
        self._colorbar(img, d, x, self.W - m - x)
        return np.asarray(img)

    def end_card(self, lines_left, lines_right, last_frame):
        s = self.s
        base = Image.fromarray(last_frame).convert('RGBA')
        base = Image.alpha_composite(base, Image.new('RGBA', base.size, BG + (232,)))
        d = ImageDraw.Draw(base)
        top = (self.H / 2 / s - 225) * s
        d.text((70 * s, top), 'Summary', font=font(34 * s, True), fill=INK)
        d.text((70 * s, top + 48 * s), self.static['model_line'], font=font(14 * s), fill=MUTED)
        fk, fv = font(13 * s, True), font(19 * s, True)
        for col, lines in ((70 * s, lines_left), (self.W / 2 + 20 * s, lines_right)):
            y = top + 110 * s
            for key, val, note in lines:
                d.text((col, y), key, font=fk, fill=ACCENT)
                d.text((col, y + 20 * s), val, font=fv, fill=INK)
                if note:
                    d.text((col, y + 48 * s), note, font=font(12 * s), fill=MUTED)
                y += 92 * s
        return np.asarray(base.convert('RGB'))


class PlainCanvas:
    """Camera frame over depth, both centre-cropped to one aspect ratio. The
    only text is the distance at three probe points, sized for a 600-px-wide
    display of the video."""

    def __init__(self, width, aspect, lo, hi, lut, bg):
        self.lo, self.hi, self.lut, self.bg, self.aspect = lo, hi, lut, bg, aspect
        self.W, self.PH = even(width), even(width / aspect)
        self.gap = even(self.W / 150.0)
        self.H = 2 * self.PH + self.gap
        self.s = self.W / 600.0
        self.f_probe = font(15 * self.s, True)
        self.probes = [(0.20, 0.62), (0.50, 0.60), (0.80, 0.62)]

    def crop(self, a):
        h, w = a.shape[:2]
        cw = min(w, int(round(h * self.aspect)))
        x0 = (w - cw) // 2
        return np.ascontiguousarray(a[:, x0:x0 + cw])

    def frame(self, rgb, depth, seg, idx, n, live):
        s = self.s
        rgb, depth = self.crop(rgb), self.crop(depth)
        size = (self.W, self.PH)
        img = Image.new('RGB', (self.W, self.H), self.bg)
        img.paste(Image.fromarray(cv2.resize(rgb, size, interpolation=cv2.INTER_LANCZOS4)), (0, 0))
        img.paste(Image.fromarray(cv2.resize(colorize(depth, self.lo, self.hi, self.lut), size,
                                             interpolation=cv2.INTER_LANCZOS4)), (0, self.PH + self.gap))
        d = ImageDraw.Draw(img, 'RGBA')
        h, w = depth.shape
        for px, py in self.probes:
            yy, xx = int(py * h), int(px * w)
            dist = float(np.median(depth[yy - 2:yy + 3, xx - 2:xx + 3]))
            cx, r = px * self.W, 5 * s
            for y0 in (0, self.PH + self.gap):
                cy = y0 + py * self.PH
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255, 255), width=int(round(1.6 * s)))
                d.ellipse([cx - 1.5 * s, cy - 1.5 * s, cx + 1.5 * s, cy + 1.5 * s], fill=(255, 255, 255, 255))
            t = f'{dist:.1f} m'
            tw = d.textlength(t, font=self.f_probe)
            cy = self.PH + self.gap + py * self.PH
            d.rounded_rectangle([cx - tw / 2 - 7 * s, cy - 36 * s, cx + tw / 2 + 7 * s, cy - 11 * s],
                                radius=5 * s, fill=(0, 0, 0, 175))
            d.text((cx - tw / 2, cy - 33 * s), t, font=self.f_probe, fill=(255, 255, 255))
        return np.asarray(img)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--segment', action='append', required=True,
                   help='KITTI drive, optionally drive:start:end (frame indices); repeat for several')
    p.add_argument('--frames-root', default=os.path.join(ROOT, 'datasets/raw_data/kitti/kitti_video'))
    p.add_argument('--gt-root', default=os.path.join(ROOT, 'datasets/raw_data/kitti/val'))
    p.add_argument('--calib-root', default=os.path.join(ROOT, 'datasets/raw_data/kitti/kitti_rgb_jpg'))
    p.add_argument('--out', help='output .mp4 (stats go next to it as .json)')
    p.add_argument('--no-video', action='store_true', help='measure only')
    p.add_argument('--stats-out', help='stats json path for --no-video')
    p.add_argument('--input-size', type=int, default=518)
    p.add_argument('--amp', choices=list(AMP), default='fp16')
    p.add_argument('--jpeg-quality', type=int, default=0,
                   help='re-encode each frame as JPEG at this quality before the model (0 = off)')
    p.add_argument('--withhold-k', action='store_true')
    p.add_argument('--slow-pre', action='store_true', help='reference CPU preprocessing (model.image2tensor)')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--scale', type=float, default=1.5, help='output size relative to the 1242-px frame')
    p.add_argument('--depth-min', type=float, default=3.0)
    p.add_argument('--depth-max', type=float, default=80.0)
    p.add_argument('--crf', type=int, default=18)
    p.add_argument('--layout', choices=['full', 'plain'], default='full')
    p.add_argument('--plain-width', type=int, default=1200)
    p.add_argument('--plain-aspect', type=float, default=16 / 9, help='width / height of each panel')
    p.add_argument('--plain-bg', default='0b1f4b', help='hex colour of the strip between the panels')
    p.add_argument('--preview', default='', help='comma-separated frame indices of the video to save as PNG too')
    p.add_argument('--model-line', default='Depth Anything V2 (ViT-L) with a camera-intrinsics encoder · '
                                           'trained on NYUv2 + KITTI + Virtual KITTI 2 + Cityscapes')
    args = p.parse_args()
    assert args.no_video or args.out, 'give --out or --no-video'

    device = 'cuda'
    torch.backends.cudnn.benchmark = True
    ns = argparse.Namespace(checkpoint=args.checkpoint, released=None, encoder='vitl')
    model, uses_k, meta = load_model(ns, device)
    assert not meta.get('relative'), 'this demo draws metric depth; give a metric checkpoint'
    feed_k = uses_k and not args.withhold_k
    amp_dtype = AMP[args.amp]
    params_m = sum(q.numel() for q in model.parameters()) / 1e6
    gpu = torch.cuda.get_device_name(0)
    static = {'model_line': args.model_line, 'gpu_short': gpu.replace('NVIDIA GeForce ', ''),
              'precision': {'none': 'FP32', 'fp16': 'FP16 autocast', 'bf16': 'BF16 autocast'}[args.amp],
              'params_m': params_m}
    segments = [Segment(sp, args) for sp in args.segment]
    print(f"model {meta['checkpoint']} | K: {feed_k} | amp {args.amp} | {params_m:.1f} M params | {gpu}", flush=True)

    canvas = writer = None
    if not args.no_video:
        if args.layout == 'plain':
            canvas = PlainCanvas(args.plain_width, args.plain_aspect, args.depth_min, args.depth_max, spectral_lut(),
                                 tuple(int(args.plain_bg[j:j + 2], 16) for j in (0, 2, 4)))
        else:
            canvas = Canvas(args.scale, args.depth_min, args.depth_max, spectral_lut(), static)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        writer = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', f'{canvas.W}x{canvas.H}', '-r', f'{CAMERA_HZ / args.stride:g}', '-i', '-',
             '-c:v', 'libx264', '-preset', 'slow', '-crf', str(args.crf), '-pix_fmt', 'yuv420p',
             '-movflags', '+faststart', args.out], stdin=subprocess.PIPE)

    total = sum(len(s.frames) for s in segments)
    stats, all_t, all_m, last, k = {'segments': []}, [], [], None, 0
    for seg in segments:
        seg_t, seg_m, recent, shape, net_hw = [], [], [], None, None
        # warm-up, not timed: cudnn autotunes per input shape and drives differ in size
        warm = cv2.imread(seg.frames[0], cv2.IMREAD_COLOR)
        for _ in range(15):
            infer(model, warm, seg.K if feed_k else None, args.input_size, amp_dtype, device,
                  args.slow_pre, args.jpeg_quality)
        for i, path in enumerate(seg.frames):
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            depth, net_hw, t = infer(model, bgr, seg.K if feed_k else None, args.input_size, amp_dtype, device,
                                     args.slow_pre, args.jpeg_quality)
            shape = depth.shape
            seg_t.append(t)
            recent = (recent + [t])[-20:]
            gt, m = seg.gt(path), None
            if gt is not None and gt.shape == depth.shape:
                valid = (gt > 1e-3) & (gt < 80.0)
                if valid.sum() >= 10:
                    m = depth_metrics(depth[valid], gt[valid])
                    m['scale'] = float(np.median(depth[valid]) / np.median(gt[valid]))
                    seg_m.append(m)
            if writer is not None:
                r = np.mean(recent, axis=0)
                live = {'net_h': net_hw[0], 'net_w': net_hw[1], 'fps': 1.0 / r.sum(), 'ms': 1e3 * r.sum(),
                        'net_ms': 1e3 * r[2],
                        'abs_rel': m and m['abs_rel'], 'd1': m and m['d1'], 'rmse': m and m['rmse'],
                        'run_abs_rel': float(np.mean([q['abs_rel'] for q in seg_m])) if seg_m else None}
                last = canvas.frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth, seg, k, total, live)
                writer.stdin.write(last.tobytes())
                if str(k) in args.preview.split(','):
                    Image.fromarray(last).save(os.path.splitext(args.out)[0] + f'_preview{k:04d}.png')
            k += 1
            if k % 100 == 0:
                print(f'{k}/{total}', flush=True)
        T = np.asarray(seg_t)
        entry = {'drive': seg.drive, 'frames': len(seg.frames), 'image_hw': list(shape), 'network_hw': list(net_hw),
                 'K': {'fx': seg.K[0, 0], 'fy': seg.K[1, 1], 'cx': seg.K[0, 2], 'cy': seg.K[1, 2]},
                 'ms_jpeg': 1e3 * T[:, 0].mean(), 'ms_pre': 1e3 * T[:, 1].mean(), 'ms_net': 1e3 * T[:, 2].mean(),
                 'ms_post': 1e3 * T[:, 3].mean(),
                 'ms_end_to_end': 1e3 * T.sum(1).mean(), 'ms_end_to_end_p95': 1e3 * np.percentile(T.sum(1), 95),
                 'fps_end_to_end': 1.0 / T.sum(1).mean(), 'fps_network_only': 1.0 / T[:, 2].mean(),
                 'frames_with_lidar': len(seg_m)}
        for key in ('abs_rel', 'd1', 'rmse', 'silog', 'scale'):
            entry[key] = float(np.mean([q[key] for q in seg_m])) if seg_m else None
        stats['segments'].append(entry)
        all_t += seg_t
        all_m += seg_m
        print(json.dumps(entry), flush=True)

    T = np.asarray(all_t)
    stats['overall'] = {
        'checkpoint': meta['checkpoint'], 'epoch': meta.get('epoch'), 'feeds_K': feed_k, 'params_million': params_m,
        'gpu': gpu, 'precision': static['precision'], 'input_size': args.input_size,
        'jpeg_quality': args.jpeg_quality, 'frames': int(len(T)), 'torch': torch.__version__,
        'slow_pre': args.slow_pre,
        'ms_jpeg': 1e3 * T[:, 0].mean(), 'ms_pre': 1e3 * T[:, 1].mean(), 'ms_net': 1e3 * T[:, 2].mean(),
        'ms_post': 1e3 * T[:, 3].mean(),
        'ms_end_to_end': 1e3 * T.sum(1).mean(), 'fps_end_to_end': 1.0 / T.sum(1).mean(),
        'fps_network_only': 1.0 / T[:, 2].mean(),
        'gpu_peak_mem_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
        'frames_with_lidar': len(all_m),
        **{key: (float(np.mean([q[key] for q in all_m])) if all_m else None)
           for key in ('abs_rel', 'd1', 'rmse', 'silog', 'scale')}}
    o = stats['overall']

    n_card = int(6 * CAMERA_HZ / args.stride) if args.layout == 'full' else 0
    if writer is not None and n_card:
        seg0 = stats['segments'][0]
        left = [
            ('CAMERA RESOLUTION', ' / '.join(sorted({f"{e['image_hw'][1]} × {e['image_hw'][0]} px" for e in stats['segments']})),
             f"network input {seg0['network_hw'][1]} × {seg0['network_hw'][0]} px (short side {args.input_size}, patch 14)"),
            ('SPEED, END-TO-END', f"{o['fps_end_to_end']:.1f} FPS   ({o['ms_end_to_end']:.0f} ms per frame)",
             f"prepare frame {o['ms_jpeg'] + o['ms_pre']:.0f} ms · network {o['ms_net']:.0f} ms · "
             f"upsample + download {o['ms_post']:.0f} ms"),
            ('AGAINST THE CAMERA', f"{o['fps_end_to_end'] / CAMERA_HZ:.1f}× real time",
             f"the camera records {CAMERA_HZ:g} frames per second; this video plays at that rate"),
            ('HARDWARE', f"{static['gpu_short']} · {static['precision']}",
             f"{params_m:.0f} M parameters · peak GPU memory {o['gpu_peak_mem_gb']:.1f} GB · batch 1"),
        ]
        right = [
            ('FRAMES PROCESSED', f"{o['frames']:,}   ({o['frames'] * args.stride / CAMERA_HZ:.0f} s of driving)",
             ', '.join(dict.fromkeys(s.label.replace('KITTI raw · ', '') for s in segments))
             + ' · drives not used in training'),
        ]
        if all_m:
            right += [
                ('ERROR vs LiDAR (AbsRel)', f"{o['abs_rel']:.3f}",
                 f"mean over {o['frames_with_lidar']:,} frames with annotated depth, 0–80 m, no scale alignment"),
                ('δ1 ACCURACY / RMSE', f"{o['d1']:.3f}   /   {o['rmse']:.2f} m",
                 f"median predicted / true depth ratio {o['scale']:.3f}"),
            ]
        right.append(('OUTPUT', 'metric depth in metres, per pixel',
                      'each frame predicted on its own from one image and the camera matrix K'))
        card = canvas.end_card(left, right, last)
        for _ in range(n_card):
            writer.stdin.write(card.tobytes())
        Image.fromarray(card).save(os.path.splitext(args.out)[0] + '_summary.png')
    if writer is not None:
        writer.stdin.close()
        rc = writer.wait()
        assert rc == 0, f'ffmpeg exit {rc}'
        stats['video'] = {'path': os.path.abspath(args.out), 'width': canvas.W, 'height': canvas.H,
                          'fps': CAMERA_HZ / args.stride, 'seconds': (total + n_card) * args.stride / CAMERA_HZ}

    out_json = args.stats_out or (os.path.splitext(args.out)[0] + '.json' if args.out else None)
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(stats, f, indent=1)
    print(json.dumps(o, indent=1))
    print('VIDEO_DONE', flush=True)


if __name__ == '__main__':
    main()
