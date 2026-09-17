#!/usr/bin/env python3
"""Generate one representative center point for each tree crown."""
import argparse
import geopandas as gpd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpkg", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layer", default="tree_crowns")
    args = p.parse_args()

    crowns = gpd.read_file(args.gpkg, layer=args.layer)
    crowns = crowns[crowns.geometry.notna() & ~crowns.geometry.is_empty].copy()
    crowns["geometry"] = crowns.geometry.representative_point()
    crowns.to_file(args.output, driver="GPKG", layer="tree_centers")
    print(f"Saved {len(crowns)} tree centers to {args.output}")


if __name__ == "__main__":
    main()
