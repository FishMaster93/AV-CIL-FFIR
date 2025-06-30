# gen_fish_audio_features.py
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from model.MobileNetv2 import MobileNetV2
import torch
import torchaudio
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Fish', choices=['Fish'])
args = parser.parse_args()

model = MobileNetV2()
pretrained_dict = torch.load('../model/pretrained/MobileNetV2_pretrained.pth', map_location='cpu')['model']
model.load_state_dict(pretrained_dict, strict=False)
model.eval()

# Fish dataset configuration
data_root = '/mnt/fast/nobackup/scratch4weeks/mc02229/Fish_dataset'
audio_root = os.path.join(data_root, 'audio_dataset')

# Get all audio files
audio_files = glob.glob(os.path.join(audio_root, '*', '*', '*', '*.wav'))
print(f"Found {len(audio_files)} audio files")

audio_pretrained_feature_dict = {}

for i, audio_path in enumerate(audio_files):
    print(f"{i}/{len(audio_files)}: {os.path.basename(audio_path)}")

    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform - waveform.mean()

    # Compute mel spectrogram (128 mel bins, 128x128 as mentioned in paper)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=64000,
        n_fft=2048,
        hop_length=1024,
        n_mels=128,
        window_fn=torch.hann_window
    )

    mel_spectrogram = mel_transform(waveform)

    # Convert to log scale
    mel_spectrogram = torch.log(mel_spectrogram + 1e-8)

    # Resize to 128x128 as mentioned in paper
    mel_spectrogram = torch.nn.functional.interpolate(
        mel_spectrogram.unsqueeze(0),
        size=(128, 128),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    mel_spectrogram = mel_spectrogram.unsqueeze(dim=0).unsqueeze(dim=0)
    with torch.no_grad():
        out = model(mel_spectrogram).squeeze(dim=0).detach().numpy()

    v_id = os.path.splitext(os.path.basename(audio_path))[0]
    audio_pretrained_feature_dict[v_id] = out

save_dir = os.path.join(data_root, 'audio_pretrained_feature')
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'audio_pretrained_feature_dict.npy'), audio_pretrained_feature_dict)

print(f"Saved {len(audio_pretrained_feature_dict)} audio features")