#!/usr/bin/env python3
"""Train a Vision Transformer on per-tree CHM + hyperspectral chips."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import timm
except ImportError as exc:
    raise SystemExit("Install timm first: pip install timm") from exc


class TreeSpeciesDataset(Dataset):
    def __init__(self, records, root, size=224):
        self.records = records.reset_index(drop=True)
        self.root = Path(root)
        self.size = size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        path = self.root / row["image_name"]
        with rasterio.open(path) as src:
            x = src.read().astype(np.float32)
            x[~np.isfinite(x)] = 0
            for c in range(x.shape[0]):
                band = x[c]
                lo, hi = np.percentile(band, [2, 98]) if np.any(band != 0) else (0, 1)
                if hi > lo:
                    x[c] = np.clip((band - lo) / (hi - lo), 0, 1)
        x = torch.from_numpy(x)
        x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(self.size, self.size), mode="bilinear", align_corners=False).squeeze(0)
        return x, int(row["label"])


def get_label_col(df):
    for c in ["taxonID", "species", "species_name", "taxon", "target_id"]:
        if c in df.columns:
            return c
    raise ValueError("attributes.csv must contain taxonID/species/species_name/taxon/target_id")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", required=True)
    p.add_argument("--output", default="species_vit.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="vit_base_patch16_224")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sample_dir = Path(args.sample_dir)
    meta = pd.read_csv(sample_dir / "attributes.csv")
    meta = meta[meta["image_name"].map(lambda x: (sample_dir / str(x)).exists())].copy()
    label_col = get_label_col(meta)
    meta[label_col] = meta[label_col].astype(str)
    classes = sorted(meta[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    meta["label"] = meta[label_col].map(class_to_idx)

    train_df, val_df = train_test_split(meta, test_size=args.val_size, random_state=args.seed,
                                        stratify=meta["label"] if len(classes) > 1 else None)
    train_ds = TreeSpeciesDataset(train_df, sample_dir)
    val_ds = TreeSpeciesDataset(val_df, sample_dir)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    with rasterio.open(sample_dir / str(meta.iloc[0]["image_name"])) as src:
        in_chans = src.count

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=False, in_chans=in_chans, num_classes=len(classes))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                ps.extend(pred.tolist())
                ys.extend(y.numpy().tolist())
        acc = accuracy_score(ys, ps)
        f1 = f1_score(ys, ps, average="macro", zero_division=0)
        print(f"Epoch {epoch+1}/{args.epochs}: val_accuracy={acc:.4f} val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "classes": classes, "in_chans": in_chans,
                        "model_name": args.model}, args.output)

    with open(str(args.output) + ".classes.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Best model saved to {args.output}")


if __name__ == "__main__":
    main()
