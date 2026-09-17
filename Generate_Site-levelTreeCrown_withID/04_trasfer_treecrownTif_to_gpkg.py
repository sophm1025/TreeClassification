#!/usr/bin/env python3
"""Convert a labeled crown raster into a GeoPackage."""
import argparse
import os

import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tif", required=True)
    p.add_argument("--gpkg", required=True)
    p.add_argument("--layer", default="tree_crowns")
    p.add_argument("--min-area", type=float, default=4.0)
    args = p.parse_args()

    with rasterio.open(args.tif) as src:
        arr = src.read(1)
        feats = []
        for geom, value in shapes(arr, mask=arr > 0, transform=src.transform):
            poly = shape(geom)
            if poly.area >= args.min_area:
                feats.append({"tree_id": int(value), "geometry": poly})
        gdf = gpd.GeoDataFrame(feats, crs=src.crs)
    os.makedirs(os.path.dirname(os.path.abspath(args.gpkg)), exist_ok=True)
    gdf.to_file(args.gpkg, layer=args.layer, driver="GPKG")
    print(f"Saved {len(gdf)} crowns to {args.gpkg}")


if __name__ == "__main__":
    main()
