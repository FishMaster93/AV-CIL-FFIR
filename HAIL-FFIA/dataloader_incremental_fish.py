import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import random
import h5py


class FishLoader(Dataset):
    def __init__(self, args, mode='train', modality='visual', incremental_step=0):
        self.mode = mode
        self.args = args
        self.modality = modality
        self.incremental_step = incremental_step

        # Fish dataset configuration
        self.data_root = '/mnt/fast/nobackup/scratch4weeks/mc02229/Fish_dataset'

        # Fish species order (each species is one incremental step)
        self.fish_types = ['Red_Tilapia', 'Tilapia', 'Jade_Perch', 'Black_Perch', 'Lotus_Carp', 'Sunfish']
        self.intensity_levels = ['None', 'Weak', 'Medium', 'Strong']

        # Load pre-extracted features
        self.load_features()

        # Generate data splits
        self.generate_splits()

        # Set current step data
        self.set_current_step_data()

    def load_features(self):
        """Load pre-extracted audio and visual features"""
        # Load audio features
        audio_feature_path = os.path.join(self.data_root, 'audio_pretrained_feature',
                                          'audio_pretrained_feature_dict.npy')
        if os.path.exists(audio_feature_path):
            self.all_audio_pretrained_features = np.load(audio_feature_path, allow_pickle=True).item()
        else:
            self.all_audio_pretrained_features = {}

        # Load visual features - try both .npy and .h5 formats
        visual_feature_npy_path = os.path.join(self.data_root, 'visual_pretrained_feature',
                                               'visual_pretrained_feature_dict.npy')
        visual_feature_h5_path = os.path.join(self.data_root, 'visual_pretrained_feature', 'visual_features.h5')

        if os.path.exists(visual_feature_npy_path):
            self.all_visual_pretrained_features = np.load(visual_feature_npy_path, allow_pickle=True).item()
        elif os.path.exists(visual_feature_h5_path):
            self.all_visual_pretrained_features = h5py.File(visual_feature_h5_path, 'r')
        else:
            self.all_visual_pretrained_features = {}

    def generate_splits(self):
        """Generate train/val/test splits"""
        # Create category mappings like original code
        self.all_id_category_dict = {'train': {}, 'val': {}, 'test': {}}
        self.all_classId_vid_dict = {'train': {}, 'val': {}, 'test': {}}

        # Initialize class ID to video ID mapping
        for split in ['train', 'val', 'test']:
            for class_id in range(4):  # 4 intensity levels
                self.all_classId_vid_dict[split][class_id] = []

        # Get all audio-video file pairs
        audio_root = os.path.join(self.data_root, 'audio_dataset')

        for fish_type in self.fish_types:
            for intensity in self.intensity_levels:
                # Get audio files for this fish-intensity combination
                audio_pattern = os.path.join(audio_root, '*', fish_type, intensity, '*.wav')
                audio_files = glob.glob(audio_pattern)

                # Create data entries for files that have both audio and visual features
                valid_files = []
                for audio_file in audio_files:
                    file_id = os.path.splitext(os.path.basename(audio_file))[0]

                    # Check if both audio and visual features exist
                    has_audio = file_id in self.all_audio_pretrained_features
                    has_visual = file_id in self.all_visual_pretrained_features

                    if has_audio and has_visual:
                        valid_files.append(file_id)

                # Split files into train/val/test (70%/10%/20% like original paper)
                random.seed(42)  # Fixed seed for reproducible splits
                random.shuffle(valid_files)

                n_total = len(valid_files)
                if n_total == 0:
                    continue

                n_train = int(n_total * 0.7)
                n_val = int(n_total * 0.1)

                train_files = valid_files[:n_train]
                val_files = valid_files[n_train:n_train + n_val]
                test_files = valid_files[n_train + n_val:]

                intensity_id = self.intensity_levels.index(intensity)

                # Store in dictionaries like original code
                for file_id in train_files:
                    self.all_id_category_dict['train'][file_id] = intensity
                    self.all_classId_vid_dict['train'][intensity_id].append(file_id)

                for file_id in val_files:
                    self.all_id_category_dict['val'][file_id] = intensity
                    self.all_classId_vid_dict['val'][intensity_id].append(file_id)

                for file_id in test_files:
                    self.all_id_category_dict['test'][file_id] = intensity
                    self.all_classId_vid_dict['test'][intensity_id].append(file_id)

    def set_current_step_data(self):
        """Set data for current incremental step"""
        current_fish = self.fish_types[self.incremental_step]    # current learning fish

        if self.mode == 'train':
            split_dict = self.all_id_category_dict['train']
        elif self.mode == 'val':
            split_dict = self.all_id_category_dict['val']
        else:  # test
            split_dict = self.all_id_category_dict['test']

        if self.mode == 'train' or self.mode == 'val':
            # Training/Validation: only current fish species
            self.all_current_data_vids = []
            audio_root = os.path.join(self.data_root, 'audio_dataset')

            for intensity in self.intensity_levels:
                audio_pattern = os.path.join(audio_root, '*', current_fish, intensity, '*.wav')
                audio_files = glob.glob(audio_pattern)

                for audio_file in audio_files:
                    file_id = os.path.splitext(os.path.basename(audio_file))[0]
                    if file_id in split_dict:
                        self.all_current_data_vids.append(file_id)
        else:
            # Test: all fish species up to current step (cumulative testing)
            self.all_current_data_vids = []
            audio_root = os.path.join(self.data_root, 'audio_dataset')

            learned_fish = self.fish_types[:self.incremental_step + 1]
            for fish_type in learned_fish:
                for intensity in self.intensity_levels:
                    audio_pattern = os.path.join(audio_root, '*', fish_type, intensity, '*.wav')
                    audio_files = glob.glob(audio_pattern)

                    for audio_file in audio_files:
                        file_id = os.path.splitext(os.path.basename(audio_file))[0]
                        if file_id in split_dict:
                            self.all_current_data_vids.append(file_id)

    def set_incremental_step(self, step):
        """Update incremental step"""
        self.incremental_step = step
        self.set_current_step_data()

    def __getitem__(self, index):
        vid = self.all_current_data_vids[index]

        # Get category information
        if self.mode == 'train':
            category = self.all_id_category_dict['train'][vid]
        elif self.mode == 'val':
            category = self.all_id_category_dict['val'][vid]
        else:
            category = self.all_id_category_dict['test'][vid]

        category_id = self.intensity_levels.index(category)

        # Load features like original code
        if 'visual' in self.modality:
            if hasattr(self.all_visual_pretrained_features, 'keys'):  # h5 file
                visual_feature = self.all_visual_pretrained_features[vid][()]
            else:  # numpy dict
                visual_feature = self.all_visual_pretrained_features[vid]
            visual_feature = torch.Tensor(visual_feature)

        if 'audio' in self.modality:
            audio_feature = self.all_audio_pretrained_features[vid]
            audio_feature = torch.Tensor(audio_feature)

        # Return based on modality
        if self.modality == 'visual':
            return visual_feature, category_id
        elif self.modality == 'audio':
            return audio_feature, category_id
        else:  # audio-visual
            return (visual_feature, audio_feature), category_id

    def close_visual_features_h5(self):
        """Close h5 file if opened"""
        if hasattr(self.all_visual_pretrained_features, 'close'):
            self.all_visual_pretrained_features.close()

    def __len__(self):
        return len(self.all_current_data_vids)