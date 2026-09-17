#!/usr/bin/env python3
"""Create per-tree CHM + HSI training chips."""
import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shp", required=True)
    p.add_argument("--hsi", required=True)
    p.add_argument("--chm", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--first-n-bands", type=int, default=None)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(args.shp)
    idcol = "individual" if "individual" in gdf.columns else next(
        (c for c in ["individualID", "tree_id", "id"] if c in gdf.columns), None)
    if idcol is None:
        raise ValueError("Training shapefile needs an individual/tree ID column.")

    records = []
    with rasterio.open(args.hsi) as hsi, rasterio.open(args.chm) as chm:
        if hsi.crs != gdf.crs:
            gdf = gdf.to_crs(hsi.crs)
        if chm.count != 1:
            raise ValueError("CHM raster must have one band.")
        bands = list(range(1, hsi.count + 1))
        if args.first_n_bands:
            bands = bands[:args.first_n_bands]

        for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Generating samples"):
            geom = [mapping(row.geometry)]
            try:
                hsi_img, transform = mask(hsi, geom, crop=True, nodata=hsi.nodata, indexes=bands)
                chm_img, chm_transform = mask(chm, geom, crop=True, nodata=chm.nodata)
            except ValueError:
                continue

            # Reproject/resample CHM to HSI chip if needed.
            if chm_img.shape[1:] != hsi_img.shape[1:]:
                from rasterio.warp import reproject, Resampling
                resized = np.zeros((1, hsi_img.shape[1], hsi_img.shape[2]), dtype=np.float32)
                reproject(
                    source=chm_img.astype(np.float32), destination=resized,
                    src_transform=chm_transform,
                    src_crs=chm.crs, dst_transform=transform, dst_crs=hsi.crs,
                    resampling=Resampling.bilinear,
                )
                chm_img = resized

            combined = np.concatenate([chm_img.astype(np.float32), hsi_img.astype(np.float32)], axis=0)
            name = str(row[idcol]).replace("/", "_")
            out_path = out / f"{name}.tif"
            profile = hsi.profile.copy()
            profile.update(driver="GTiff", count=combined.shape[0], dtype="float32",
                           height=combined.shape[1], width=combined.shape[2], transform=transform,
                           compress="lzw", nodata=None)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(combined)

            rec = row.drop(labels="geometry").to_dict()
            rec["image_name"] = out_path.name
            rec["target_id"] = rec.get("taxonID", rec.get("species", ""))
            records.append(rec)

    pd.DataFrame(records).to_csv(out / "attributes.csv", index=False)
    print(f"Saved {len(records)} samples to {out}")


if __name__ == "__main__":
    main()
