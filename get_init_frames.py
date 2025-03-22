import h5py
import cv2 
from tqdm import tqdm
import argparse 
import os
"""
- Script to get first image/video from each camera view. Useful for 
  specifying pixels.
0fdae376_blackcup_plate_coffee   25d59628_brush_pan_clay  b9c595b1_knife_plate_clay
1202ba72_shotglass_plate_coffee  a8dfe5f4_grayduster      d347da5c_bluescooper_bowl_nuts
"""


parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, help="output directory for images")
args = parser.parse_args()

data = h5py.File(f"{args.out_dir}/data00000000.h5", "r")
imgs = data["imgs"]
os.makedirs(args.out_dir, exist_ok=True)
os.makedirs(f"{args.out_dir}/views", exist_ok=True)
for idx in tqdm(range(imgs.shape[1])):
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{args.out_dir}/views/view_{idx}_vid.mp4", fourcc, fps, (imgs.shape[3], imgs.shape[2]))
    for img in imgs[:, idx, :, :, :]:
        out.write(img)
    out.release()
    cv2.imwrite(f"{args.out_dir}/views/view_{idx}_img.png", imgs[0, idx])
    