import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import random
import h5py


class RedTilapiaLoader(Dataset):
    def __init__(self, args, mode='train', modality='visual'):
        self.mode = mode
        self.args = args
        self.modality = modality

        # Fish dataset configuration - 只使用Red Tilapia
        self.data_root = '/mnt/fast/nobackup/scratch4weeks/mc02229/Fish_dataset'
        self.fish_type = 'Red_Tilapia'  # 固定为Red Tilapia
        self.intensity_levels = ['None', 'Weak', 'Medium', 'Strong']

        # Load pre-extracted features
        self.load_features()

        # Generate data splits for Red Tilapia only
        self.generate_splits()

        # Set current data
        self.set_current_data()

    def load_features(self):
        """Load pre-extracted audio and visual features"""
        # Load audio features
        audio_feature_path = os.path.join(self.data_root, 'audio_pretrained_feature',
                                          'audio_pretrained_feature_dict.npy')
        if os.path.exists(audio_feature_path):
            self.all_audio_pretrained_features = np.load(audio_feature_path, allow_pickle=True).item()
        else:
            self.all_audio_pretrained_features = {}
            print(f"Warning: Audio features not found at {audio_feature_path}")

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
            print(f"Warning: Visual features not found")

    def generate_splits(self):
        """Generate train/val/test splits for Red Tilapia only"""
        self.all_id_category_dict = {'train': {}, 'val': {}, 'test': {}}
        self.all_classId_vid_dict = {'train': {}, 'val': {}, 'test': {}}

        # Initialize class ID to video ID mapping
        for split in ['train', 'val', 'test']:
            for class_id in range(4):  # 4 intensity levels
                self.all_classId_vid_dict[split][class_id] = []

        # Get all audio-video file pairs for Red Tilapia only
        audio_root = os.path.join(self.data_root, 'audio_dataset')

        print(f"Loading data for {self.fish_type}...")

        for intensity in self.intensity_levels:
            # Get audio files for Red Tilapia with this intensity
            audio_pattern = os.path.join(audio_root, '*', self.fish_type, intensity, '*.wav')
            audio_files = glob.glob(audio_pattern)

            print(f"Found {len(audio_files)} {intensity} intensity files")

            # Create data entries for files that have both audio and visual features
            valid_files = []
            for audio_file in audio_files:
                file_id = os.path.splitext(os.path.basename(audio_file))[0]

                # Check if both audio and visual features exist
                has_audio = file_id in self.all_audio_pretrained_features
                has_visual = file_id in self.all_visual_pretrained_features

                if has_audio and has_visual:
                    valid_files.append(file_id)
                elif not has_audio:
                    print(f"Missing audio feature for {file_id}")
                elif not has_visual:
                    print(f"Missing visual feature for {file_id}")

            print(f"Valid files for {intensity}: {len(valid_files)}")

            # Split files into train/val/test (70%/15%/15% split)
            random.seed(42)  # Fixed seed for reproducible splits
            random.shuffle(valid_files)

            n_total = len(valid_files)
            if n_total == 0:
                print(f"Warning: No valid files found for {intensity} intensity")
                continue

            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.15)

            train_files = valid_files[:n_train]
            val_files = valid_files[n_train:n_train + n_val]
            test_files = valid_files[n_train + n_val:]

            intensity_id = self.intensity_levels.index(intensity)

            # Store in dictionaries
            for file_id in train_files:
                self.all_id_category_dict['train'][file_id] = intensity
                self.all_classId_vid_dict['train'][intensity_id].append(file_id)

            for file_id in val_files:
                self.all_id_category_dict['val'][file_id] = intensity
                self.all_classId_vid_dict['val'][intensity_id].append(file_id)

            for file_id in test_files:
                self.all_id_category_dict['test'][file_id] = intensity
                self.all_classId_vid_dict['test'][intensity_id].append(file_id)

        # Print split statistics
        for split in ['train', 'val', 'test']:
            total_files = len(self.all_id_category_dict[split])
            print(f"{split.upper()} split: {total_files} files")
            for i, intensity in enumerate(self.intensity_levels):
                count = len(self.all_classId_vid_dict[split][i])
                print(f"  {intensity}: {count} files")

    def set_current_data(self):
        """Set data for current split"""
        if self.mode == 'train':
            split_dict = self.all_id_category_dict['train']
        elif self.mode == 'val':
            split_dict = self.all_id_category_dict['val']
        else:  # test
            split_dict = self.all_id_category_dict['test']

        self.all_current_data_vids = list(split_dict.keys())
        print(f"{self.mode.upper()} dataset: {len(self.all_current_data_vids)} samples")

    def get_class_distribution(self):
        """Get class distribution for current split"""
        if self.mode == 'train':
            class_vid_dict = self.all_classId_vid_dict['train']
        elif self.mode == 'val':
            class_vid_dict = self.all_classId_vid_dict['val']
        else:
            class_vid_dict = self.all_classId_vid_dict['test']

        distribution = {}
        for i, intensity in enumerate(self.intensity_levels):
            distribution[intensity] = len(class_vid_dict[i])

        return distribution

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

        # Load features
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

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close_visual_features_h5()


# Test function to verify dataloader works
def test_dataloader(args):
    """Test the Red Tilapia dataloader"""
    print("Testing Red Tilapia DataLoader...")

    # Test all modalities
    modalities = ['visual', 'audio', 'audio-visual']

    for modality in modalities:
        print(f"\n--- Testing {modality} modality ---")

        # Create datasets
        train_dataset = RedTilapiaLoader(args, mode='train', modality=modality)
        val_dataset = RedTilapiaLoader(args, mode='val', modality=modality)
        test_dataset = RedTilapiaLoader(args, mode='test', modality=modality)

        # Print class distributions
        print("Train distribution:", train_dataset.get_class_distribution())
        print("Val distribution:", val_dataset.get_class_distribution())
        print("Test distribution:", test_dataset.get_class_distribution())

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        # Test first batch
        try:
            for data, labels in train_loader:
                print(f"Batch data type: {type(data)}")
                if modality == 'audio-visual':
                    visual_data, audio_data = data
                    print(f"Visual shape: {visual_data.shape}, Audio shape: {audio_data.shape}")
                else:
                    print(f"Data shape: {data.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Label values: {labels}")
                break
        except Exception as e:
            print(f"Error testing {modality}: {e}")

        # Cleanup
        train_dataset.close_visual_features_h5()
        val_dataset.close_visual_features_h5()
        test_dataset.close_visual_features_h5()


if __name__ == "__main__":
    # Simple test
    class Args:
        def __init__(self):
            pass


    args = Args()
    test_dataloader(args)