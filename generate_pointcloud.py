import numpy as np
import open3d as o3d 
import cv2 
import h5py 
import yaml

def depth2xyzmap(depth, K, uvs=None):
  invalid_mask = (depth<0.001)
  H,W = depth.shape[:2]
  if uvs is None:
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:,0]
    vs = uvs[:,1]
  zs = depth[vs,us]
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = np.zeros((H,W,3), dtype=np.float32)
  xyz_map[vs,us] = pts
  xyz_map[invalid_mask] = 0
  return xyz_map

camera_info = yaml.safe_load(open("/svl/u/tarunc/camera_ext_calibration.yaml", "r"))

f = h5py.File("/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/0c78ea75/data00000000.h5", "r")
depths = np.array(f['depths']).astype(np.float64) / 1e3
pcds = [depth2xyzmap(depths[0, i], np.asarray(camera_info[i]['color_intrinsic_matrix'])) for i in range(8)]

"""
Results from this file 
- get segmentation from each frame 
- np array containing per-timestep pointclouds 
"""
breakpoint()