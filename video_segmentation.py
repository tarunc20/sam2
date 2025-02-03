import torch
from sam2.build_sam import build_sam2_video_predictor
import h5py
from tqdm import tqdm
import cv2
import numpy as np
import os 
import argparse 
import cv2
import yaml 
import numpy as np

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

def create_video_from_frames(frames, video_name):
    """
    Create a video from a list of frames (NumPy arrays).
    
    Parameters:
    frames (list): List of frames as NumPy arrays
    video_name (str): Name of the output video file (include .mp4 extension)
    
    Returns:
    None
    """
    if not frames:
        print("Error: No frames provided")
        return
    
    # Infer size from the first frame
    size = frames[0].shape[1::-1]
    
    # Fixed fps
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, fps, size)
    
    for frame in frames:
        # Ensure the frame is in the correct format (uint8)
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # If the frame is grayscale, convert to RGB
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # Ensure all frames have the same size as the first frame
        if frame.shape[1::-1] != size:
            frame = cv2.resize(frame, size)
        
        out.write(frame)
    
    out.release()
    print(f"Video saved as {video_name}")

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

parser = argparse.ArgumentParser()
parser.add_argument("--h5_file", type=str)
parser.add_argument("--out_dir", type=str)
parser.add_argument("-x", "--x_coords", nargs='+', type=int)
parser.add_argument("-y", "--y_coords", nargs='+', type=int)
parser.add_argument("--t", type=str)
args = parser.parse_args()
#assert(len(args.x_coords) == len(args.y_coords)), "Must have the same number of x and y coords"
pixels = [[x, y] for x, y in zip(args.x_coords, args.y_coords)]
data = h5py.File(args.h5_file, "r")


# make this an argument 
camera_info = yaml.safe_load(open("/viscam/projects/robotool/videos_0121//camera_ext_calibration_0121.yaml", "r"))
hf_pcds = h5py.File(f"{args.out_dir}/sam2_pcds_{args.t}_updated.h5", "w")
hf_masks = h5py.File(f"{args.out_dir}/sam2_masks_{args.t}_updated.h5", "w")
os.makedirs(f"{args.out_dir}/sam_{args.t}", exist_ok=True)
for cam in range(8):
    pcd_group = hf_pcds.create_group(f"cam_{cam}")
    os.makedirs(f"{args.out_dir}/sam_imgs_{cam}_{args.t}", exist_ok=True)
    for i in tqdm(range(len(data['imgs']))):
        assert(cv2.imwrite(f"{args.out_dir}/sam_imgs_{cam}_{args.t}/{i + 10000}.jpg", data['imgs'][i, cam, :, :, :]))
    video_dir = f"{args.out_dir}/sam_imgs_{cam}_{args.t}"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)
        ann_frame_idx = 0
        ann_obj_id = 255
        points = np.array([pixels[cam]], dtype=np.float32)
        labels = np.array([255], dtype=np.int32)
        out_masks = []
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        for _, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            out_masks.append((out_mask_logits[0, 0] > 0.0).cpu().numpy())
        imgs = []
        out_masks = np.array(out_masks)
        for i, f in enumerate(sorted(os.listdir(video_dir))):
            imgs.append(cv2.imread(os.path.join(video_dir, f)))   
            imgs[i] = imgs[i].astype(np.float64) / 255.
            imgs[i][out_masks[i]] = imgs[i][out_masks[i]] * 0.5 + np.array((1, 0, 0))[None, None, :] * 0.5
            imgs[i] = (imgs[i] * 255).astype(np.uint8)

        all_pcds = [depth2xyzmap(data['depths'][i, cam], np.asarray(camera_info[cam]["color_intrinsic_matrix"]))[out_masks[i]] 
                        for i in range(len(out_masks))]
        for i in range(len(all_pcds)):
            pcd_group.create_dataset(f"pcd_{i}", data=all_pcds[i].astype(np.float32))
        out_masks = out_masks.astype(np.uint8) * 255
        hf_masks.create_dataset(f"masks_{cam}", data=out_masks)
        #np.save(f"{args.out_dir}")
        create_video_from_frames(imgs, f"{args.out_dir}/sam_{args.t}/test_{cam}_{args.t}.mp4")
    for f in os.listdir(f"{args.out_dir}/sam_imgs_{cam}_{args.t}"):
        os.remove(os.path.join(f"{args.out_dir}/sam_imgs_{cam}_{args.t}", f))
    os.rmdir(f"{args.out_dir}/sam_imgs_{cam}_{args.t}")
hf_pcds.close()
hf_masks.close()