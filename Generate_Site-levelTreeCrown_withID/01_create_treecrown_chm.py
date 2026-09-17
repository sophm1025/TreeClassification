#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from skimage import feature, segmentation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--site', required=True)
    p.add_argument('--year', type=int, default=2022)
    p.add_argument('--root', default='/storage/group/tvq5043/default/Zhuohong')
    p.add_argument('--min-height', type=float, default=3.0)
    p.add_argument('--peak-height', type=float, default=5.0)
    p.add_argument('--min-distance', type=int, default=2)
    args = p.parse_args()

    root = Path(args.root)
    candidates = [
        root/'NEON_data'/'CHM_MLBS_intersection.tif',
        root/'NEON_data'/'original_NEON_Data'/f'{args.site}{args.year}'/f'CHM_{args.site}_intersection.tif',
        root/'NEON_data'/'original_NEON_Data'/f'{args.site}{args.year}'/'CHM_intersection.tif',
    ]
    src_path = next((x for x in candidates if x.exists()), None)
    if src_path is None:
        raise FileNotFoundError('Could not find CHM raster. Checked:\n' + '\n'.join(map(str, candidates)))

    out_dir = root/'NEON_data'/'Preprocess'/args.site
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/'CHM_based_tree_crown_above3m.tif'

    with rasterio.open(src_path) as src:
        chm = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    chm[chm < args.min_height] = 0
    smooth = gaussian_filter(chm, sigma=0.5)
    peaks = feature.peak_local_max(smooth, min_distance=args.min_distance, threshold_abs=args.peak_height)
    markers = np.zeros_like(smooth, dtype=np.int32)
    for i, (r, c) in enumerate(peaks, 1):
        markers[r, c] = i
    labels = segmentation.watershed(-smooth, markers, mask=smooth >= args.peak_height)

    profile.update(dtype='int32', count=1, nodata=0, compress='lzw')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(labels.astype(np.int32), 1)
    print(f'Saved rough crowns to {out_path}')

if __name__ == '__main__':
    main()
