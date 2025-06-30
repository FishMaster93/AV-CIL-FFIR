#!/usr/bin/env python3
"""
Audio-Visual Backbone Training Script following the exact paper methodology
严格按照论文实现的Audio-Visual骨干网络训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from dataloader_step1 import RedTilapiaLoader


class AudioVisualBackbone(nn.Module):
    """
    Audio-Visual Backbone following the exact paper methodology
    Implements bidirectional cross-modal attention mechanism as described in paper
    """

    def __init__(self, args, visual_dim=1028, audio_dim=1024, feature_dim=512):
        super(AudioVisualBackbone, self).__init__()
        self.args = args
        self.feature_dim = feature_dim

        # Input feature projection to unified dimension d
        self.visual_input_proj = nn.Linear(visual_dim, feature_dim)
        self.audio_input_proj = nn.Linear(audio_dim, feature_dim)

        # Learnable projection matrices W^a, W^v ∈ R^{d×d}
        self.W_audio = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_visual = nn.Linear(feature_dim, feature_dim, bias=False)

        # Final fusion projection matrices U^v, U^a ∈ R^{d×d}
        self.U_visual = nn.Linear(feature_dim, feature_dim, bias=False)
        self.U_audio = nn.Linear(feature_dim, feature_dim, bias=False)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)
        )

    def forward(self, visual, audio):
        """
        Args:
            visual: [batch, L, S, visual_dim] 或 [batch, n, visual_dim] 或 [batch, visual_dim]
            audio: [batch, audio_dim]
        Returns:
            logits: [batch, 4]
            similarity_loss: 相似性损失
        """
        batch_size = visual.shape[0]

        # Process visual feature dimensions, convert to [batch, L, S, d]
        if visual.dim() == 2:
            # [batch, visual_dim] → [batch, 1, 1, visual_dim]
            visual = visual.unsqueeze(1).unsqueeze(1)
            L, S = 1, 1
        elif visual.dim() == 3:
            # [batch, n, visual_dim] → [batch, n, 1, visual_dim]
            visual = visual.unsqueeze(2)
            L, S = visual.shape[1], 1
        else:
            # [batch, L, S, visual_dim]
            L, S = visual.shape[1], visual.shape[2]

        # Project to unified feature dimension
        # f^v ∈ R^{L×S×d}, f^a ∈ R^{d}
        f_visual = self.visual_input_proj(visual)  # [batch, L, S, d]
        f_audio = self.audio_input_proj(audio)  # [batch, d]

        # 1. Compute bidirectional attention scores
        # Score^a = tanh(f^a * W^a)
        score_audio = torch.tanh(self.W_audio(f_audio))  # [batch, d]

        # Score^v_l = tanh(f^v_l * W^v) for each frame l
        f_visual_reshaped = f_visual.view(batch_size * L, S, self.feature_dim)
        score_visual_frames = torch.tanh(self.W_visual(f_visual_reshaped))  # [batch*L, S, d]
        score_visual_frames = score_visual_frames.view(batch_size, L, S, self.feature_dim)  # [batch, L, S, d]

        # 2. Spatial-temporal attention computation (Visual Stream)
        enhanced_visual_frames = []
        spatial_weighted_scores = []

        for l in range(L):
            # Get scores for frame l
            score_v_l = score_visual_frames[:, l, :, :]  # [batch, S, d]
            f_v_l = f_visual[:, l, :, :]  # [batch, S, d]

            # Compute spatial attention weights
            # w^{Spa.}_l = Softmax(Score^a ⊙ Score^v_l)
            score_audio_expanded = score_audio.unsqueeze(1).expand(-1, S, -1)  # [batch, S, d]
            spatial_attention_raw = score_audio_expanded * score_v_l  # [batch, S, d] Hadamard product

            # Apply softmax after summing over feature dimension
            spatial_attention_weights = F.softmax(spatial_attention_raw.sum(dim=2), dim=1)  # [batch, S]
            spatial_attention_weights = spatial_attention_weights.unsqueeze(2)  # [batch, S, 1]

            # Score'^v_l = sum(w^{Spa.}_l ⊙ Score^v_l)
            spatially_weighted_score = (spatial_attention_weights * score_v_l).sum(dim=1)  # [batch, d]
            spatial_weighted_scores.append(spatially_weighted_score)

            # Also compute enhanced visual features sum(f^v_l ⊙ w^{Spa.}_l)
            enhanced_frame = (spatial_attention_weights * f_v_l).sum(dim=1)  # [batch, d]
            enhanced_visual_frames.append(enhanced_frame)

        # Stack all frames' spatial weighted scores and enhanced features
        spatial_weighted_scores = torch.stack(spatial_weighted_scores, dim=1)  # [batch, L, d]
        enhanced_visual_frames = torch.stack(enhanced_visual_frames, dim=1)  # [batch, L, d]

        # Compute temporal attention weights
        # w^{Tem.} = Softmax([Score'^v_1,...,Score'^v_L])
        temporal_attention_weights = F.softmax(spatial_weighted_scores.sum(dim=2), dim=1)  # [batch, L]
        temporal_attention_weights = temporal_attention_weights.unsqueeze(2)  # [batch, L, 1]

        # Compute final enhanced visual features
        # f'^v = sum^L_{l=1} w^{Tem.}_l ⊙ sum(f^v_l ⊙ w^{Spa.}_l)
        f_prime_visual = (temporal_attention_weights * enhanced_visual_frames).sum(dim=1)  # [batch, d]

        # 3. Audio attention computation (Audio Stream)
        # w^{Audio} = Softmax(1/L * sum_{l=1}^L Score^v_l ⊙ Score^a)
        avg_visual_score = score_visual_frames.mean(dim=(1, 2))  # [batch, d] average over L and S
        audio_attention_raw = avg_visual_score * score_audio  # [batch, d] Hadamard product
        audio_attention_weights = F.softmax(audio_attention_raw, dim=1)  # [batch, d]

        # Compute enhanced audio features
        # f'^a = f^a ⊙ w^{Audio}
        f_prime_audio = f_audio * audio_attention_weights  # [batch, d]

        # 4. Bidirectional similarity constraint loss
        # L_sim = 1 - (f'^v · f'^a) / (||f'^v|| · ||f'^a||)
        cosine_similarity = F.cosine_similarity(f_prime_visual, f_prime_audio, dim=1)  # [batch]
        similarity_loss = (1 - cosine_similarity).mean()  # scalar

        # 5. Final fused representation
        # f^{av} = σ(f'^v * U^v) + σ(f'^a * U^a)
        enhanced_visual_proj = torch.sigmoid(self.U_visual(f_prime_visual))  # [batch, d]
        enhanced_audio_proj = torch.sigmoid(self.U_audio(f_prime_audio))  # [batch, d]

        fused_feature = enhanced_visual_proj + enhanced_audio_proj  # [batch, d]

        # 6. Classification
        logits = self.classifier(fused_feature)  # [batch, 4]

        return logits, similarity_loss


def train_audio_visual_model(args):
    """Train Audio-Visual Backbone model"""
    print("=" * 60)
    print("Audio-Visual Backbone Training (Following Paper Methodology)")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    modality = 'audio-visual'
    train_dataset = RedTilapiaLoader(args, mode='train', modality=modality)
    val_dataset = RedTilapiaLoader(args, mode='val', modality=modality)
    test_dataset = RedTilapiaLoader(args, mode='test', modality=modality)

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.infer_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False,
                            shuffle=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.infer_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Print class distributions
    print("\nClass distributions:")
    print("Train:", train_dataset.get_class_distribution())
    print("Val:", val_dataset.get_class_distribution())
    print("Test:", test_dataset.get_class_distribution())

    # Create model
    model = AudioVisualBackbone(args, visual_dim=1028, audio_dim=1024, feature_dim=512)
    model = model.to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.lr_step_size,
                                                gamma=args.lr_gamma)

    # Training parameters
    best_val_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_class_loss = 0.0
        epoch_sim_loss = 0.0
        num_steps = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for data, labels in train_pbar:
            visual, audio = data
            visual = visual.to(device)
            audio = audio.to(device)
            labels = labels.to(device)

            # Forward pass
            logits, similarity_loss = model(visual, audio)

            # Classification loss
            classification_loss = F.cross_entropy(logits, labels)

            # Total loss
            total_loss = classification_loss + args.sim_loss_weight * similarity_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            epoch_train_loss += total_loss.item()
            epoch_class_loss += classification_loss.item()
            epoch_sim_loss += similarity_loss.item()
            num_steps += 1

            # Update progress bar
            train_pbar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'class_loss': f'{classification_loss.item():.4f}',
                'sim_loss': f'{similarity_loss.item():.4f}'
            })

        # Average losses
        epoch_train_loss /= num_steps
        epoch_class_loss /= num_steps
        epoch_sim_loss /= num_steps

        # Validation phase
        model.eval()
        all_val_logits = []
        all_val_labels = []
        val_similarities = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for val_data, val_labels in val_pbar:
                val_visual, val_audio = val_data
                val_visual = val_visual.to(device)
                val_audio = val_audio.to(device)

                val_logits, val_similarity_loss = model(val_visual, val_audio)

                val_logits = F.softmax(val_logits, dim=-1).cpu()
                all_val_logits.append(val_logits)
                all_val_labels.append(val_labels)
                val_similarities.append(val_similarity_loss.item())

        all_val_logits = torch.cat(all_val_logits, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)
        val_acc = (all_val_logits.argmax(dim=1) == all_val_labels).float().mean().item()

        avg_similarity = np.mean(val_similarities)

        # Update learning rate
        scheduler.step()

        print(f'Epoch {epoch + 1}: train_loss={epoch_train_loss:.5f}, '
              f'class_loss={epoch_class_loss:.5f}, sim_loss={epoch_sim_loss:.5f}, '
              f'val_acc={val_acc:.6f}, avg_similarity={avg_similarity:.4f}, '
              f'lr={scheduler.get_last_lr()[0]:.6f}')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f'./save/RedTilapia/audio_visual/best_backbone_model.pkl'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save complete model
            torch.save(model, save_path)

            # Save checkpoint
            checkpoint_path = f'./save/RedTilapia/audio_visual/backbone_checkpoint.pkl'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'args': args
            }, checkpoint_path)

            print(f'*** Saved best backbone: {val_acc:.6f} ***')

    # Test evaluation
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)

    # Load best model for testing
    model = torch.load(save_path, map_location=device)
    model.eval()

    all_test_logits = []
    all_test_labels = []

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for test_data, test_labels in test_pbar:
            test_visual, test_audio = test_data
            test_visual = test_visual.to(device)
            test_audio = test_audio.to(device)

            test_logits, _ = model(test_visual, test_audio)

            test_logits = F.softmax(test_logits, dim=-1).cpu()
            all_test_logits.append(test_logits)
            all_test_labels.append(test_labels)

    all_test_logits = torch.cat(all_test_logits, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)
    test_acc = (all_test_logits.argmax(dim=1) == all_test_labels).float().mean().item()

    print(f"\nFINAL RESULTS:")
    print(f"Best Validation Accuracy: {best_val_acc:.6f}")
    print(f"Test Accuracy: {test_acc:.6f}")

    # Classification report
    intensity_levels = ['None', 'Weak', 'Medium', 'Strong']
    y_pred = all_test_logits.argmax(dim=1).numpy()
    y_true = all_test_labels.numpy()

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=intensity_levels))

    # Cleanup
    train_dataset.close_visual_features_h5()
    val_dataset.close_visual_features_h5()
    test_dataset.close_visual_features_h5()

    return save_path, test_acc
    epoch_sim_loss += similarity_loss.item()
    num_steps += 1

    # 更新进度条
    train_pbar.set_postfix({
        'total_loss': f'{total_loss.item():.4f}',
        'class_loss': f'{classification_loss.item():.4f}',
        'sim_loss': f'{similarity_loss.item():.4f}'
    })

    # 平均损失


    epoch_train_loss /= num_steps
    epoch_class_loss /= num_steps
    epoch_sim_loss /= num_steps

    # 验证阶段
    model.eval()
    all_val_logits = []
    all_val_labels = []
    val_similarities = []

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validation")
        for val_data, val_labels in val_pbar:
            val_visual, val_audio = val_data
            val_visual = val_visual.to(device)
            val_audio = val_audio.to(device)

            val_logits, val_similarity_loss = model(val_visual, val_audio)

            val_logits = F.softmax(val_logits, dim=-1).cpu()
            all_val_logits.append(val_logits)
            all_val_labels.append(val_labels)
            val_similarities.append(val_similarity_loss.item())

    all_val_logits = torch.cat(all_val_logits, dim=0)
    all_val_labels = torch.cat(all_val_labels, dim=0)
    val_acc = (all_val_logits.argmax(dim=1) == all_val_labels).float().mean().item()

    avg_similarity = np.mean(val_similarities)

    # 更新学习率
    scheduler.step()

    print(f'Epoch {epoch + 1}: train_loss={epoch_train_loss:.5f}, '
          f'class_loss={epoch_class_loss:.5f}, sim_loss={epoch_sim_loss:.5f}, '
          f'val_acc={val_acc:.6f}, avg_similarity={avg_similarity:.4f}, '
          f'lr={scheduler.get_last_lr()[0]:.6f}')

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = f'./save/RedTilapia/audio_visual/best_backbone_model.pkl'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存完整模型
        torch.save(model, save_path)

        # 保存checkpoint
        checkpoint_path = f'./save/RedTilapia/audio_visual/backbone_checkpoint.pkl'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'epoch': epoch,
            'args': args
        }, checkpoint_path)

        print(f'*** Saved best backbone: {val_acc:.6f} ***')

    # 测试评估
    print("\n" + "=" * 50)
    print("EVALUATING ON TEST SET")
    print("=" * 50)

    # 加载最佳模型进行测试
    model = torch.load(save_path, map_location=device)
    model.eval()

    all_test_logits = []
    all_test_labels = []

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for test_data, test_labels in test_pbar:
            test_visual, test_audio = test_data
            test_visual = test_visual.to(device)
            test_audio = test_audio.to(device)

            test_logits, _ = model(test_visual, test_audio)

            test_logits = F.softmax(test_logits, dim=-1).cpu()
            all_test_logits.append(test_logits)
            all_test_labels.append(test_labels)

    all_test_logits = torch.cat(all_test_logits, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)
    test_acc = (all_test_logits.argmax(dim=1) == all_test_labels).float().mean().item()

    print(f"\nFINAL RESULTS:")
    print(f"Best Validation Accuracy: {best_val_acc:.6f}")
    print(f"Test Accuracy: {test_acc:.6f}")

    # 分类报告
    intensity_levels = ['None', 'Weak', 'Medium', 'Strong']
    y_pred = all_test_logits.argmax(dim=1).numpy()
    y_true = all_test_labels.numpy()

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=intensity_levels))

    # 清理
    train_dataset.close_visual_features_h5()
    val_dataset.close_visual_features_h5()
    test_dataset.close_visual_features_h5()

    return save_path, test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio-Visual Backbone Training')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--infer_batch_size', type=int, default=64,
                        help='Inference batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lr_step_size', type=int, default=30,
                        help='Learning rate scheduler step size')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='Learning rate scheduler gamma')
    parser.add_argument('--sim_loss_weight', type=float, default=0.1,
                        help='Weight for similarity constraint loss')

    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"Arguments: {args}")

    # Train model
    model_path, test_accuracy = train_audio_visual_model(args)

    print(f"\nTraining completed!")
    print(f"Best backbone saved at: {model_path}")
    print(f"Final test accuracy: {test_accuracy:.6f}")