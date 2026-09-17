#!/usr/bin/env python3
"""Evaluate predicted species against labeled reference points/polygons."""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--prediction-column", default="pred_species")
    p.add_argument("--truth-column", default=None)
    p.add_argument("--tolerance", type=float, default=2.0)
    p.add_argument("--output", default="evaluation")
    args = p.parse_args()

    pred = gpd.read_file(args.predictions)
    truth = gpd.read_file(args.truth)
    truth_col = args.truth_column or next((c for c in ["taxonID", "species", "species_name", "taxon"] if c in truth.columns), None)
    if truth_col is None:
        raise ValueError("Could not find truth species column.")
    if args.prediction_column not in pred.columns:
        raise ValueError(f"Missing prediction column {args.prediction_column}")

    pred = pred.to_crs(truth.crs)
    joined = gpd.sjoin_nearest(pred[[args.prediction_column, "geometry"]], truth[[truth_col, "geometry"]],
                                how="inner", max_distance=args.tolerance, distance_col="distance")
    y_true = joined[truth_col].astype(str)
    y_pred = joined[args.prediction_column].astype(str)

    metrics = {
        "n_matches": int(len(joined)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(report).T.to_csv(str(out) + "_classification_report.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(str(out) + "_confusion_matrix.csv")
    joined.to_file(str(out) + "_matched.gpkg", layer="matched", driver="GPKG")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
