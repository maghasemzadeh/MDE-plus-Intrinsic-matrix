"""Summarise the camera-descriptor ablation (tools/multidomain/run_descriptor_ablation.sh).

For every arm: AbsRel and d1 per test set, the mean over the eight thesis sets, the number of
sets where the arm is better than the 9-D re-run (paired Wilcoxon p<0.05 on per-image AbsRel),
and the K-sweep slope d log(depth) / d log(f) from the k=0.7 and k=1.4 runs on NYU+KITTI
(200 images each; 1 = depth follows the focal length as geometry says, 0 = K ignored).
"""
import json
import math
import os

from scipy.stats import wilcoxon

R = 'results/multidomain'
SETS = ['nyu', 'kitti', 'cityscapes', 'diode_outdoor',
        'sunrgbd_kv1', 'sunrgbd_kv2', 'sunrgbd_realsense', 'sunrgbd_xtion']
ARMS = [('full9 (reported 9-D, re-run)', 'descabl_full9', 'descabl_full9'),
        ('9-D original run (md2s_P1)', 'md2s_P1', 'descabl_P1'),
        ('fov2: FoV only (DA3-style)', 'descabl_fov2', 'descabl_fov2'),
        ('k4: fx,fy,cx,cy only', 'descabl_k4', 'descabl_k4'),
        ('focal2: fx,fy only', 'descabl_focal2', 'descabl_focal2'),
        ('noshape: 9-D minus aspect,area', 'descabl_noshape', 'descabl_noshape'),
        ('nofov: 9-D minus FoV angles', 'descabl_nofov', 'descabl_nofov'),
        ('plus11: 9-D + log fx, log fy', 'descabl_plus11', 'descabl_plus11'),
        ('no K (control md2s_C)', 'md2s_C', None)]
REF = 'descabl_full9'


def per_image(run, s):
    p = os.path.join(R, run, s + '.json')
    if not os.path.exists(p):
        return None
    return {m['id']: m for m in json.load(open(p))['per_image']}


def summary(run):
    p = os.path.join(R, run, 'summary.json')
    return json.load(open(p))['results'] if os.path.exists(p) else None


def slope(prefix):
    if prefix is None:
        return None
    a, b = summary(prefix + '_k0.7'), summary(prefix + '_k1.4')
    if not a or not b:
        return None
    vals = [(math.log(b[s]['median_scale_ratio']) - math.log(a[s]['median_scale_ratio'])) / math.log(2.0)
            for s in ('nyu', 'kitti')]
    return sum(vals) / len(vals)


ref = {s: per_image(REF, s) for s in SETS}
rows = []
for name, run, kp in ARMS:
    res = summary(run)
    if not res:
        print(f'{name}: missing'); continue
    absr = [res[s]['abs_rel'] for s in SETS]
    d1 = [res[s]['d1'] for s in SETS]
    better = worse = 0
    if run != REF:
        for s in SETS:
            a, b = per_image(run, s), ref[s]
            ids = sorted(set(a) & set(b))
            x = [a[i]['abs_rel'] for i in ids]; y = [b[i]['abs_rel'] for i in ids]
            if len(ids) < 10 or x == y:
                continue
            p = wilcoxon(x, y).pvalue
            if p < 0.05:
                if sum(x) < sum(y):
                    better += 1
                else:
                    worse += 1
    rows.append((name, absr, d1, better, worse, slope(kp)))

hdr = 'arm'.ljust(34) + ''.join(s.replace('sunrgbd_', '')[:8].rjust(9) for s in SETS) + '   meanAR  meanD1  vs9D(+/-)  Kslope'
print(hdr)
for name, absr, d1, b, w, k in rows:
    print(name.ljust(34) + ''.join(f'{v:9.4f}' for v in absr)
          + f'   {sum(absr)/len(absr):.4f}  {sum(d1)/len(d1):.4f}   {b:2d}/{w:<2d}     '
          + (f'{k:.2f}' if k is not None else '  -'))
