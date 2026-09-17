#!/usr/bin/env python3
"""Run the trained Vision Transformer on tree-center points."""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
import timm


def normalize_chip(x):
    x = x.astype(np.float32)
    x[~np.isfinite(x)] = 0
    for c in range(x.shape[0]):
        band = x[c]
        nz = band != 0
        if nz.any():
            lo, hi = np.percentile(band[nz], [2, 98])
            if hi > lo:
                x[c] = np.clip((band - lo) / (hi - lo), 0, 1)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--centers", required=True)
    p.add_argument("--hsi", required=True)
    p.add_argument("--chm", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--window-m", type=float, default=8.0)
    args = p.parse_args()

    bundle = torch.load(args.model, map_location="cpu")
    classes = bundle["classes"]
    model_name = bundle.get("model_name", "vit_base_patch16_224")
    with rasterio.open(args.hsi) as hsi, rasterio.open(args.chm) as chm:
        gdf = gpd.read_file(args.centers)
        gdf = gdf.to_crs(hsi.crs) if gdf.crs != hsi.crs else gdf
        model = timm.create_model(model_name, pretrained=False, in_chans=hsi.count + 1, num_classes=len(classes))
        model.load_state_dict(bundle["model"])
        model.eval()
        transform = hsi.transform
        px = abs(transform.a)
        half_px = max(1, int((args.window_m / 2) / px))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        preds, probs = [], []
        for geom in gdf.geometry:
            r, c = rasterio.transform.rowcol(transform, geom.x, geom.y)
            row0, col0 = r - half_px, c - half_px
            size = 2 * half_px
            win = rasterio.windows.Window(col0, row0, size, size)
            hsi_chip = hsi.read(window=win, boundless=True, fill_value=0)
            chm_chip = chm.read(1, window=win, boundless=True, fill_value=0)[None, ...]
            x = normalize_chip(np.concatenate([chm_chip, hsi_chip], axis=0))
            x = torch.from_numpy(x).unsqueeze(0)
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).to(device)
            with torch.no_grad():
                p = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            idx = int(np.argmax(p))
            preds.append(classes[idx])
            probs.append(float(p[idx]))

    out = gdf.copy()
    out["pred_species"] = preds
    out["confidence"] = probs
    out.to_file(args.output, layer="predictions", driver="GPKG")
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
