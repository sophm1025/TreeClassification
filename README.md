# NeON tree mapping

## Repository structure

```text
NEON-TreeMapping/
├── 00_GEE_download_ref002_RGB_CHM.py
├── Generate_Site-levelTreeCrown_withID/
│   ├── 01_create_treecrown_chm.py
│   ├── 02_refine_treecrown_DL_rgb.py
│   ├── 03_01_predict_treecrown.py
│   ├── 03_02_merge_patch_to_site.py
│   └── 04_trasfer_treecrownTif_to_gpkg.py
└── Generate_train_pool/
    ├── 00_generate_training_sample.py
    ├── 03_generate_samplepool.py
    ├── 05_train.py
    ├── 06_generate_center_point_from_gpkg.py
    ├── 07_inference_full_site.py
    └── 08_evaluate.py
```

## Workflow

1. Download RGB/CHM data with `00_GEE_download_ref002_RGB_CHM.py`.
2. Run the scripts in `Generate_Site-levelTreeCrown_withID/` to create rough CHM crowns, refine them with Mask R-CNN and RGB, merge patch predictions, and convert the final raster crowns to GeoPackage.
3. Run `Generate_train_pool/00_generate_training_sample.py` to prepare field samples.
4. Run `Generate_train_pool/03_generate_samplepool.py` to create CHM + hyperspectral training TIFFs.
5. Run `Generate_train_pool/05_train.py` to train the Vision Transformer species classifier.
6. Run `Generate_train_pool/06_generate_center_point_from_gpkg.py` and `07_inference_full_site.py` for full-site species inference.
7. Run `Generate_train_pool/08_evaluate.py` for evaluation.

> The GEE downloader is intentionally left as a project-specific placeholder because the original repository states that this script still requires modification.
