#!/usr/bin/env python3
import argparse, os
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tqdm import tqdm


def main():
    p=argparse.ArgumentParser(); p.add_argument('--rgb',required=True); p.add_argument('--weights',required=True)
    p.add_argument('--out-dir',required=True); p.add_argument('--patch',type=int,default=800); p.add_argument('--overlap',type=int,default=128)
    p.add_argument('--score',type=float,default=0.5); args=p.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=maskrcnn_resnet50_fpn(weights=None,num_classes=2); model.load_state_dict(torch.load(args.weights,map_location=device)); model.to(device).eval()
    with rasterio.open(args.rgb) as src:
        step=args.patch-args.overlap
        for r,row in enumerate(tqdm(range(0,src.height,step),desc='Rows')):
            for c,col in enumerate(range(0,src.width,step)):
                h=min(args.patch,src.height-row); w=min(args.patch,src.width-col)
                arr=src.read(window=Window(col,row,w,h)).astype(np.float32)/255.0
                x=torch.from_numpy(arr[:3]).unsqueeze(0).to(device)
                with torch.no_grad(): out=model(x)[0]
                mask_out=np.zeros((h,w),dtype=np.uint32); oid=1
                for score,m in zip(out['scores'].detach().cpu().numpy(),out['masks'].detach().cpu().numpy()):
                    if score<args.score: continue
                    mm=(m[0,:h,:w]>0.5)
                    if mm.sum()==0: continue
                    mask_out[mm]=oid; oid+=1
                profile=src.profile.copy(); profile.update(height=h,width=w,count=1,dtype='uint32',compress='lzw')
                with rasterio.open(os.path.join(args.out_dir,f'patch_{r}_{c}.tif'),'w',**profile) as dst: dst.write(mask_out,1)

if __name__=='__main__': main()
