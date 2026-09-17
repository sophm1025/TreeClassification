# NEON Tree Mapping

An automated workflow for mapping individual tree crowns and classifying tree species using NEON remote sensing and vegetation structure data.

The pipeline combines **Canopy Height Models (CHM)**, **high-resolution RGB imagery**, and **hyperspectral imagery (HSI)** to delineate individual tree crowns and classify their species. Initial tree crowns are generated from CHM data, refined using a **Mask R-CNN** trained on RGB imagery, and then classified by species using a **Vision Transformer (ViT)** trained on hyperspectral and canopy-height information.

---

## Overview

The workflow consists of two main components:

1. **Individual Tree Crown Mapping**
   - Generate approximate tree crowns from CHM data.
   - Train a Mask R-CNN using RGB imagery and the initial crown boundaries.
   - Predict refined tree crowns across an entire NEON site.
   - Merge overlapping prediction tiles.
   - Convert the final crown raster into geospatial tree-crown polygons.

2. **Tree Species Classification**
   - Process NEON field vegetation data.
   - Generate georeferenced training points.
   - Extract CHM and hyperspectral data around individual trees.
   - Train a Vision Transformer for species classification.
   - Generate center points for detected tree crowns.
   - Run the trained model across the site.
   - Evaluate species-classification accuracy.

---

## Pipeline

```text
NEON Remote Sensing + Field Data
            |
            |--- Canopy Height Model (CHM)
            |--- RGB Imagery
            |--- Hyperspectral Imagery (HSI)
            |--- Vegetation Structure Data
            |
            v
Generate Initial Tree Crowns from CHM
            |
            v
Train Mask R-CNN with RGB Imagery
            |
            v
Predict Refined Tree Crowns
            |
            v
Merge Prediction Tiles
            |
            v
Individual Tree Crown Map
            |
            v
Generate Species Training Samples
            |
            v
Combine CHM + Hyperspectral Information
            |
            v
Train Vision Transformer
            |
            v
Species Classification
            |
            v
Accuracy Evaluation
```

---

## Repository Structure

```text
NEON-TreeMapping/
│
├── README.md
├── requirements_species_vit.txt
│
├── Generate_Site-levelTreeCrown_withID/
│   │
│   ├── 01_create_treecrown_chm.py
│   ├── 02_refine_treecrown_DL_rgb.py
│   ├── 03_01_predict_treecrown.py
│   ├── 03_02_merge_patch_to_site.py
│   ├── 04_trasfer_treecrownTif_to_gpkg.py
│   └── README.md
│
└── Generate_train_pool/
    │
    ├── 00_generate_training_sample.py
    ├── 03_generate_samplepool.py
    ├── 05_train.py
    ├── 06_generate_center_point_from_gpkg.py
    ├── 07_inference_full_site.py
    ├── 08_evaluate.py
    └── README.md
```

---

# Data Requirements

The processing pipeline assumes that the required NEON datasets have already been downloaded.

For each site and year, the workflow requires:

- **Canopy Height Model (CHM)**
- **High-resolution RGB imagery**
- **Hyperspectral imagery (HSI)**
- **NEON vegetation structure / field inventory data**

Data acquisition is treated as a prerequisite rather than part of the core mapping pipeline.

The pipeline begins once these datasets are available locally.

A typical data organization is:

```text
data/
└── SITE_YEAR/
    ├── CHM.tif
    ├── RGB.tif
    ├── HSI.tif
    └── vegetation/
```

For example:

```text
data/
└── MLBS_2022/
    ├── CHM.tif
    ├── RGB.tif
    ├── HSI.tif
    └── vegetation/
```

---

# Part 1: Individual Tree Crown Mapping

The first part of the pipeline generates individual tree crown boundaries.

## Step 1: Generate Initial Tree Crowns from CHM

Script:

```text
Generate_Site-levelTreeCrown_withID/01_create_treecrown_chm.py
```

The Canopy Height Model provides the estimated height of vegetation above the ground.

The script:

1. Removes low vegetation.
2. Smooths the CHM.
3. Detects local canopy-height maxima representing potential treetops.
4. Uses the detected treetops as markers.
5. Applies watershed segmentation.
6. Assigns a unique integer ID to each candidate tree crown.

Conceptually:

```text
CHM
 |
 v
Height Filtering
 |
 v
Gaussian Smoothing
 |
 v
Treetop Detection
 |
 v
Watershed Segmentation
 |
 v
Initial Tree Crown Raster
```

Example:

```bash
python Generate_Site-levelTreeCrown_withID/01_create_treecrown_chm.py \
    --site MLBS \
    --year 2022
```

The resulting crown boundaries provide approximate labels for training the RGB-based crown-refinement model.

---

## Step 2: Refine Tree Crowns with Mask R-CNN

Script:

```text
Generate_Site-levelTreeCrown_withID/02_refine_treecrown_DL_rgb.py
```

CHM resolution is relatively coarse compared with NEON RGB imagery. Therefore, the initial crown boundaries are refined using high-resolution RGB imagery.

This stage uses **Mask R-CNN**, an instance-segmentation neural network.

Mask R-CNN simultaneously predicts:

- tree locations,
- bounding boxes,
- and pixel-level crown masks.

The CHM-derived crown raster acts as the initial training label source.

The script also handles spatial alignment between the crown raster and RGB imagery before model training.

Example:

```bash
python Generate_Site-levelTreeCrown_withID/02_refine_treecrown_DL_rgb.py \
    --site MLBS \
    --year 2022
```

Model checkpoints are saved during training.

---

## Step 3: Predict Tree Crowns Across the Site

Script:

```text
Generate_Site-levelTreeCrown_withID/03_01_predict_treecrown.py
```

Large NEON RGB mosaics cannot generally be passed through the Mask R-CNN model as a single image.

The site is therefore divided into overlapping image patches.

```text
Full RGB Mosaic
      |
      v
Overlapping Patches
      |
      v
Mask R-CNN
      |
      v
Tree Crown Masks
```

Each predicted tree crown receives a local ID within its patch.

The overlapping patches reduce boundary artifacts and make inference possible on large imagery.

Example:

```bash
python Generate_Site-levelTreeCrown_withID/03_01_predict_treecrown.py \
    --site MLBS \
    --year 2022
```

---

## Step 4: Merge Crown Prediction Patches

Script:

```text
Generate_Site-levelTreeCrown_withID/03_02_merge_patch_to_site.py
```

Because neighboring patches overlap, the same tree may appear in multiple predictions.

The merging script compares crowns in overlapping regions using **Intersection over Union (IoU)**.

```text
IoU = Intersection Area / Union Area
```

Crowns exceeding the configured IoU threshold are treated as the same tree.

A global tree ID is then assigned so that each detected crown has a unique identifier across the site.

Example:

```bash
python Generate_Site-levelTreeCrown_withID/03_02_merge_patch_to_site.py \
    --site MLBS \
    --year 2022
```

Output:

```text
Site-level crown raster
```

where:

```text
0 = background
1 = tree crown 1
2 = tree crown 2
3 = tree crown 3
...
```

---

## Step 5: Convert Tree Crowns to GeoPackage

Script:

```text
Generate_Site-levelTreeCrown_withID/04_trasfer_treecrownTif_to_gpkg.py
```

The site-level raster is converted into vector polygons.

Each polygon represents an individual tree crown and retains its unique tree ID.

Output:

```text
tree_crowns.gpkg
```

This produces a standard GIS-compatible representation that can be opened using tools such as:

- QGIS
- ArcGIS
- GeoPandas

---

# Part 2: Generate Species Training Data

After tree crowns have been mapped, the second portion of the pipeline prepares training data for species classification.

## Step 6: Generate Training Samples

Script:

```text
Generate_train_pool/00_generate_training_sample.py
```

NEON vegetation structure data provides field observations for individual trees, including information such as:

- tree coordinates,
- species/taxon ID,
- tree height,
- crown diameter,
- and individual identifiers.

The training-sample workflow converts these observations into geospatial training locations and filters samples for use in species classification.

Example:

```bash
python Generate_train_pool/00_generate_training_sample.py \
    --site MLBS \
    --year 2022
```

---

## Step 7: Generate CHM + Hyperspectral Training Pool

Script:

```text
Generate_train_pool/03_generate_samplepool.py
```

For each selected field tree, a local image sample is extracted.

The training samples combine:

```text
CHM
+
Hyperspectral Bands
```

The CHM contributes structural information about the tree, while hyperspectral imagery contributes detailed spectral information related to vegetation characteristics.

Each tree becomes a multi-channel image sample.

Conceptually:

```text
Field Tree Location
        |
        v
Extract Local Region
        |
        +---- CHM
        |
        +---- Hyperspectral Bands
        |
        v
Multi-channel Tree Sample
        |
        v
Species Label
```

These samples form the training dataset for the Vision Transformer.

---

# Part 3: Vision Transformer Species Classification

## Step 8: Train the Vision Transformer

Script:

```text
Generate_train_pool/05_train.py
```

Species classification is performed using a **Vision Transformer (ViT)**.

Unlike a conventional convolutional neural network, a Vision Transformer divides an image into patches and processes the patches using transformer attention.

Conceptually:

```text
Tree Image
    |
    v
Image Patches
    |
    v
Patch Embeddings
    |
    v
Transformer Encoder
    |
    v
Classification Head
    |
    v
Tree Species
```

The model uses the CHM + hyperspectral training samples generated in the previous stage.

The classifier predicts the NEON taxon/species associated with each tree sample.

Example:

```bash
python Generate_train_pool/05_train.py \
    --site MLBS \
    --year 2022
```

Training produces:

- model checkpoints,
- class mappings,
- training statistics,
- and the final trained model.

---

# Part 4: Full-Site Species Mapping

## Step 9: Generate Tree Crown Center Points

Script:

```text
Generate_train_pool/06_generate_center_point_from_gpkg.py
```

The individual tree crown polygons generated earlier are converted to representative center points.

```text
Tree Crown Polygon
        |
        v
Representative Point
        |
        v
Tree Location
```

These points are used to extract remote-sensing information for each detected tree.

Example:

```bash
python Generate_train_pool/06_generate_center_point_from_gpkg.py \
    --site MLBS \
    --year 2022
```

---

## Step 10: Run Full-Site Species Inference

Script:

```text
Generate_train_pool/07_inference_full_site.py
```

The trained Vision Transformer is applied to detected trees across the entire site.

For each tree:

1. Locate the crown center.
2. Extract the corresponding CHM and hyperspectral sample.
3. Apply the same preprocessing used during training.
4. Pass the sample through the trained Vision Transformer.
5. Obtain predicted species probabilities.
6. Assign the highest-probability species to the tree.

Conceptually:

```text
Detected Tree Crown
        |
        v
Extract CHM + HSI
        |
        v
Vision Transformer
        |
        v
Species Probabilities
        |
        v
Predicted Species
```

The result is a site-level map containing individual tree crowns and their predicted species.

Example:

```bash
python Generate_train_pool/07_inference_full_site.py \
    --site MLBS \
    --year 2022
```

---

# Part 5: Model Evaluation

## Step 11: Evaluate Species Classification

Script:

```text
Generate_train_pool/08_evaluate.py
```

The final stage evaluates predictions against known species labels.

Evaluation includes metrics such as:

- overall accuracy,
- precision,
- recall,
- F1 score,
- per-species performance,
- and confusion matrix.

The confusion matrix helps identify which species the model commonly confuses.

Example:

```bash
python Generate_train_pool/08_evaluate.py \
    --site MLBS \
    --year 2022
```

---

# Complete Workflow

The full processing workflow is:

```text
1. Download / obtain NEON data
             |
             v
2. CHM crown delineation
             |
             v
3. Mask R-CNN RGB crown refinement
             |
             v
4. Full-site Mask R-CNN prediction
             |
             v
5. Merge overlapping crown predictions
             |
             v
6. Convert crown raster to GeoPackage
             |
             v
7. Process NEON vegetation observations
             |
             v
8. Generate CHM + HSI training samples
             |
             v
9. Train Vision Transformer
             |
             v
10. Generate tree crown center points
             |
             v
11. Run full-site species inference
             |
             v
12. Evaluate classification accuracy
```

---

# Installation

A Python virtual environment is recommended.

```bash
python -m venv .venv
```

Activate the environment.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements_species_vit.txt
```

Major dependencies include:

- PyTorch
- torchvision
- timm
- NumPy
- pandas
- Rasterio
- GeoPandas
- Shapely
- scikit-image
- scikit-learn
- SciPy
- tqdm

A CUDA-compatible GPU is strongly recommended for Mask R-CNN and Vision Transformer training.

---

# Models Used

## Mask R-CNN

**Purpose:** Individual tree crown segmentation

**Input:** High-resolution RGB imagery

**Output:** Pixel-level individual tree crown masks

Mask R-CNN is used because tree-crown mapping is an **instance segmentation** problem: the workflow must distinguish individual neighboring trees rather than simply label all tree pixels as vegetation.

---

## Vision Transformer

**Purpose:** Individual tree species classification

**Input:** Multi-channel tree samples derived from CHM and hyperspectral imagery

**Output:** Predicted species/taxon

The Vision Transformer learns relationships between spatial and spectral features within individual tree samples.

---

# Input Data Summary

| Dataset | Purpose |
|---|---|
| CHM | Initial crown delineation and tree structural information |
| RGB | High-resolution tree crown refinement |
| Hyperspectral imagery | Spectral information for species classification |
| Vegetation structure data | Species labels and field tree locations |

---

# Output Summary

The pipeline produces several intermediate and final outputs.

### Crown Mapping

```text
Initial CHM crown raster
        ↓
Mask R-CNN predictions
        ↓
Merged site-level crown raster
        ↓
Tree crown GeoPackage
```

### Species Classification

```text
CHM + HSI training samples
        ↓
Trained Vision Transformer
        ↓
Tree species predictions
        ↓
Evaluation metrics
```

The final result is a geospatial representation of individual trees containing:

- tree crown geometry,
- unique tree ID,
- predicted species,
- and model confidence where available.

---

# Notes

- The pipeline assumes the required NEON remote-sensing data has already been downloaded.
- Raster datasets should use compatible coordinate reference systems.
- CHM, RGB, and hyperspectral imagery must be spatially aligned before samples are compared or extracted.
- File paths may need to be adjusted for the local computing environment.
- GPU memory requirements depend on image size, patch size, batch size, and model configuration.
- NEON field-data column names should be checked when using different sites or data releases.
- The workflow should be validated on a complete site dataset before large-scale production use.

---

# Example Site

Many of the original scripts were developed using the NEON **MLBS (Mountain Lake Biological Station)** site.

Example:

```bash
python Generate_Site-levelTreeCrown_withID/01_create_treecrown_chm.py \
    --site MLBS \
    --year 2022
```

Other NEON sites can be processed by supplying the appropriate imagery, field data, site name, and year.

---

# Project Goal

The goal of this repository is to provide an end-to-end processing and machine-learning workflow for transforming NEON remote-sensing and field data into individual-tree species maps.

The workflow integrates:

**remote sensing + geospatial processing + instance segmentation + Vision Transformers**

to automate individual tree crown delineation and tree species classification at the site level.
