# gen_fish_visual_features.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import torch
import numpy as np
import glob
from PIL import Image
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Fish', choices=['Fish'])
args = parser.parse_args()


def load_s3d_model():

    from model.S3D import S3D
    model = S3D()
    # Load pre-trained weights
    checkpoint = torch.load('../model/pretrained/s3d_pretrained.pth', map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    return model


model = load_s3d_model()
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

# Fish dataset configuration
data_root = '/mnt/fast/nobackup/scratch4weeks/mc02229/Fish_dataset'
video_root = os.path.join(data_root, 'video_dataset')

# Get all video files
video_files = glob.glob(os.path.join(video_root, '*', '*', '*', '*.mp4'))
print(f"Found {len(video_files)} video files")

input_num_frames = 16
visual_pretrained_feature_dict = {}


def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        frame_indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        interval = total_frames / (num_frames - 1)
        frame_indices = [int(i * interval) for i in range(num_frames - 1)] + [total_frames - 1]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)  # Keep as numpy array for S3D
        else:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    cap.release()
    return frames


for i, video_path in enumerate(video_files):
    print(f"{i}/{len(video_files)}: {os.path.basename(video_path)}")

    try:
        # Extract 16 frames as mentioned in paper
        frames = extract_frames(video_path, input_num_frames)

        # Preprocess frames for S3D model
        # Resize frames to 224x224 and normalize
        processed_frames = []
        for frame in frames:
            frame = cv2.resize(frame, (224, 224))
            frame = frame.astype(np.float32) / 255.0
            # Normalize with ImageNet stats
            frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            processed_frames.append(frame)

        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        video_tensor = np.stack(processed_frames, axis=0)  # (T, H, W, C)
        video_tensor = np.transpose(video_tensor, (3, 0, 1, 2))  # (C, T, H, W)
        video_tensor = torch.from_numpy(video_tensor).unsqueeze(0)  # (1, C, T, H, W)

        if torch.cuda.is_available():
            video_tensor = video_tensor.cuda()

        with torch.no_grad():
            feature = model(video_tensor)
        feature = feature.squeeze(dim=0).detach().cpu().numpy()

        v_id = os.path.splitext(os.path.basename(video_path))[0]
        visual_pretrained_feature_dict[v_id] = feature

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        continue

save_dir = os.path.join(data_root, 'visual_pretrained_feature')
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'visual_pretrained_feature_dict.npy'), visual_pretrained_feature_dict)

print(f"Saved {len(visual_pretrained_feature_dict)} visual features")