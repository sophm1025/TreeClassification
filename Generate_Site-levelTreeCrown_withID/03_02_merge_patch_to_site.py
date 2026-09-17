#!/usr/bin/env python3
"""Merge overlapping local crown-ID patches into one site raster."""
import argparse, os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def iou(a,b):
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum(); return inter/union if union else 0.0


def main():
    p=argparse.ArgumentParser(); p.add_argument('--rgb',required=True); p.add_argument('--patch-dir',required=True); p.add_argument('--output',required=True)
    p.add_argument('--patch',type=int,default=800); p.add_argument('--overlap',type=int,default=128); p.add_argument('--iou',type=float,default=0.38); p.add_argument('--min-area',type=int,default=100); args=p.parse_args()
    step=args.patch-args.overlap
    with rasterio.open(args.rgb) as rgb:
        profile=rgb.profile.copy(); profile.update(count=1,dtype='uint32',nodata=0,compress='lzw',BIGTIFF='YES')
        H,W=rgb.height,rgb.width
        num_r=(H-1)//step+1; num_c=(W-1)//step+1
        os.makedirs(os.path.dirname(os.path.abspath(args.output)),exist_ok=True)
        with rasterio.open(args.output,'w',**profile) as dst:
            dst.write(np.zeros((H,W),dtype=np.uint32),1)
            next_id=1; maps={}; previous={}; pairs=[]
            for r in tqdm(range(num_r),desc='Merging rows'):
                for c in range(num_c):
                    pf=os.path.join(args.patch_dir,f'patch_{r}_{c}.tif')
                    if not os.path.exists(pf): continue
                    with rasterio.open(pf) as ps: arr=ps.read(1); h,w=arr.shape
                    out=np.zeros_like(arr,dtype=np.uint32)
                    for lid in np.unique(arr):
                        if lid==0 or (arr==lid).sum()<args.min_area: continue
                        gid=next_id; next_id+=1; maps[(r,c,int(lid))]=gid; m=arr==lid
                        if c>0 and (r,c-1) in previous:
                            for pgid,pm in previous[(r,c-1)]['right']:
                                if iou(m[:,:args.overlap], pm[:,-args.overlap:])>args.iou: pairs.append((gid,pgid))
                        if r>0 and (r-1,c) in previous:
                            for pgid,pm in previous[(r-1,c)]['bottom']:
                                if iou(m[:args.overlap,:], pm[-args.overlap:,:])>args.iou: pairs.append((gid,pgid))
                        out[m]=gid
                    win=Window(c*step,r*step,w,h); dst.write(out,1,window=win)
                    previous[(r,c)]={'right':[(int(g),out==g) for g in np.unique(out) if g],
                                         'bottom':[(int(g),out==g) for g in np.unique(out) if g]}
    # second pass: connected components over the ID-equivalence pairs
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(a,b): parent[find(a)]=find(b)
    for a,b in pairs: union(a,b)
    remap={}; n=1
    with rasterio.open(args.output,'r+') as dst:
        for r in range(num_r):
            for c in range(num_c):
                pf=os.path.join(args.patch_dir,f'patch_{r}_{c}.tif')
                if not os.path.exists(pf): continue
                with rasterio.open(pf) as ps: arr=ps.read(1); h,w=arr.shape
                out=np.zeros_like(arr,dtype=np.uint32)
                for lid in np.unique(arr):
                    gid=maps.get((r,c,int(lid)))
                    if not gid: continue
                    root=find(gid)
                    if root not in remap:
                        remap[root]=n
                        n += 1
                    out[arr==lid]=remap[root]
                dst.write(out,1,window=Window(c*step,r*step,w,h))
    print(f'Saved merged crown raster to {args.output}')


if __name__=='__main__': main()
