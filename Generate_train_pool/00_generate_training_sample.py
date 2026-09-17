#!/usr/bin/env python3
"""Prepare field training samples for the NEON tree-species model.

This combines the repository's point-shapefile step and sample-pool step into
one command, matching the README's intended Step 3 interface.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def pick_column(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {label} column. Tried: {candidates}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", required=True)
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--root", default="/storage/group/tvq5043/default/Zhuohong")
    p.add_argument("--min-height", type=float, default=5.0)
    p.add_argument("--min-crown-diameter", type=float, default=1.0)
    p.add_argument("--max-crown-diameter", type=float, default=100.0)
    p.add_argument("--sample-radius", type=float, default=2.0,
                   help="Radius in meters for each training sample polygon.")
    args = p.parse_args()

    root = Path(args.root)
    mapping_dir = root / "NEON_mapping" / args.site
    data_dir = root / "NEON_data"
    prep_dir = data_dir / "Preprocess" / args.site
    support_dir = prep_dir / "Support_data"
    sample_dir = prep_dir / "training_sample_hsi-chm"
    support_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    csv_candidates = [
        mapping_dir / f"{args.site}_tree_data.csv",
        mapping_dir / f"{args.site}_merged.csv",
        prep_dir / f"{args.site}_merged.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("No field tree CSV found. Expected one of:\n" + "\n".join(map(str, csv_candidates)))

    df = pd.read_csv(csv_path)
    xcol = pick_column(df, ["adjEasting", "easting", "x", "Easting"], "easting")
    ycol = pick_column(df, ["adjNorthing", "northing", "y", "Northing"], "northing")
    hcol = pick_column(df, ["height", "Height", "treeHeight"], "height")
    idcol = pick_column(df, ["individualID", "individual", "tree_id", "id"], "individual ID")

    diameter_col = next((c for c in ["maxCrownDiameter", "crownDiameter", "ninetyCrownDiameter"] if c in df.columns), None)

    work = df.copy()
    work = work.dropna(subset=[xcol, ycol, hcol, idcol])
    work = work[work[hcol] >= args.min_height].copy()
    if diameter_col:
        work = work.dropna(subset=[diameter_col])
        work = work[(work[diameter_col] >= args.min_crown_diameter) &
                    (work[diameter_col] <= args.max_crown_diameter)].copy()

    points = gpd.GeoDataFrame(work, geometry=[Point(x, y) for x, y in zip(work[xcol], work[ycol])],
                              crs=f"EPSG:{32600 + 17}")
    point_shp = support_dir / f"{args.site}_training_points.shp"
    points.to_file(point_shp)

    circles = points.copy()
    circles["geometry"] = circles.geometry.buffer(args.sample_radius)
    circles["individual"] = circles[idcol].astype(str)
    circle_shp = support_dir / f"{args.site}_circle.shp"
    circles.to_file(circle_shp)

    metadata_csv = sample_dir / "attributes.csv"
    metadata = pd.DataFrame(circles.drop(columns="geometry"))
    metadata.to_csv(metadata_csv, index=False)

    hsi_candidates = [
        data_dir / "original_NEON_Data" / f"{args.site}{args.year}" / f"Refl002_{args.site}_allbands_intersection.tif",
        data_dir / f"Refl002_{args.site}_allbands_intersection.tif",
    ]
    chm_candidates = [
        data_dir / "original_NEON_Data" / f"{args.site}{args.year}" / f"CHM_{args.site}_intersection.tif",
        data_dir / f"CHM_{args.site}_intersection.tif",
    ]
    hsi_path = next((p for p in hsi_candidates if p.exists()), None)
    chm_path = next((p for p in chm_candidates if p.exists()), None)
    if hsi_path is None or chm_path is None:
        raise FileNotFoundError("Could not find the HSI/CHM rasters. Edit the candidate paths in this script if your NEON files use different names.")

    sample_script = Path(__file__).with_name("03_generate_samplepool.py")
    cmd = [sys.executable, str(sample_script), "--shp", str(circle_shp), "--hsi", str(hsi_path),
           "--chm", str(chm_path), "--out", str(sample_dir)]
    subprocess.run(cmd, check=True)

    print(f"Training points: {point_shp}")
    print(f"Training polygons: {circle_shp}")
    print(f"Sample pool: {sample_dir}")


if __name__ == "__main__":
    main()
