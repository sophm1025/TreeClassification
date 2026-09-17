#!/usr/bin/env python3
"""Train the repository's Mask R-CNN crown-refinement model."""
import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.utils.data as data
import torchvision
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from tqdm import tqdm
from torchvision.models.detection import maskrcnn_resnet50_fpn


class TreePatchDataset(data.Dataset):
    def __init__(self, rgb_path, mask_path, patch_size=1024, stride=768, samples=10000):
        self.rgb = rasterio.open(rgb_path)
        self.mask = rasterio.open(mask_path)
        if (self.rgb.width, self.rgb.height) != (self.mask.width, self.mask.height):
            raise ValueError('RGB and mask must have the same raster dimensions after alignment.')
        self.patch_size = patch_size
        self.patches = [(x, y) for y in range(0, self.rgb.height - patch_size + 1, stride)
                        for x in range(0, self.rgb.width - patch_size + 1, stride)]
        np.random.shuffle(self.patches)
        self.patches = self.patches[:samples]

    def __len__(self): return len(self.patches)

    def __getitem__(self, idx):
        x, y = self.patches[idx]
        win = Window(x, y, self.patch_size, self.patch_size)
        rgb = self.rgb.read(window=win).astype(np.float32) / 255.0
        if rgb.shape[0] < 3:
            raise ValueError('RGB raster must have at least 3 bands.')
        rgb = torch.from_numpy(rgb[:3])
        mask = self.mask.read(1, window=win)
        ids = np.unique(mask); ids = ids[ids != 0]
        boxes, masks = [], []
        for oid in ids:
            m = mask == oid
            yy, xx = np.where(m)
            if len(xx) < 10: continue
            xmin, xmax, ymin, ymax = xx.min(), xx.max(), yy.min(), yy.max()
            if xmax <= xmin or ymax <= ymin: continue
            boxes.append([xmin, ymin, xmax, ymax]); masks.append(m)
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            masks = torch.tensor(np.asarray(masks), dtype=torch.uint8)
            labels = torch.ones(len(boxes), dtype=torch.int64)
        else:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            masks = torch.zeros((0,self.patch_size,self.patch_size), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'masks': masks, 'image_id': torch.tensor([idx])}
        return rgb, target


def collate(batch):
    return tuple(zip(*batch))


def align_raster(src_path, target_path, out_path, resampling):
    with rasterio.open(src_path) as src, rasterio.open(target_path) as tgt:
        profile = tgt.profile.copy()
        profile.update(count=1, dtype=src.dtypes[0], nodata=0, compress='lzw', BIGTIFF='YES')
        with rasterio.open(out_path, 'w', **profile) as dst:
            for y in range(0, tgt.height, 2048):
                for x in range(0, tgt.width, 2048):
                    h, w = min(2048, tgt.height-y), min(2048, tgt.width-x)
                    win = Window(x, y, w, h)
                    arr = np.zeros((h,w), dtype=src.dtypes[0])
                    reproject(rasterio.band(src, 1), arr, src_transform=src.transform, src_crs=src.crs,
                              dst_transform=tgt.window_transform(win), dst_crs=tgt.crs, resampling=resampling)
                    dst.write(arr, 1, window=win)


def get_model():
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, 2)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rgb', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--workdir', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--patch', type=int, default=1024)
    p.add_argument('--stride', type=int, default=768)
    p.add_argument('--samples', type=int, default=10000)
    args = p.parse_args()
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    aligned = work/'mask_aligned.tif'
    print('Aligning crown masks to RGB raster...')
    align_raster(args.mask, args.rgb, aligned, Resampling.nearest)

    ds = TreePatchDataset(args.rgb, aligned, args.patch, args.stride, args.samples)
    dl = data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model().to(device)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)

    for epoch in range(args.epochs):
        model.train(); total = 0.0; steps = 0
        for images, targets in tqdm(dl, desc=f'Epoch {epoch+1}/{args.epochs}'):
            images = [x.to(device) for x in images]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); steps += 1
        sched.step()
        ckpt = work/f'model_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), ckpt)
        print(f'epoch={epoch+1} loss={total/max(steps,1):.4f} saved={ckpt}')

if __name__ == '__main__':
    main()
