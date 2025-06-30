import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class HAIL_FFIR_Net(nn.Module):
    def __init__(self, args, step_out_class_num, incremental_step=0, pretrained_backbone_path=None):
        super(HAIL_FFIR_Net, self).__init__()
        self.args = args
        self.modality = args.modality
        self.num_classes = step_out_class_num  # Always 4 for intensity levels
        self.incremental_step = incremental_step
        self.feature_dim = 512  # Backbone feature dimension

        # Feature expansion dimension (typically an order of magnitude larger)
        self.expanded_dim = 2048  # 10x expansion as mentioned in paper

        if self.modality not in ['visual', 'audio', 'audio-visual']:
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')

        # Load pretrained backbone
        self.pretrained_backbone = None
        if pretrained_backbone_path:
            self.load_pretrained_backbone(pretrained_backbone_path)

        # Initialize backbone feature extractors
        self.init_backbone_extractors()

        # Feature expansion layers (randomly initialized and fixed)
        self.init_feature_expansion()

        # Hierarchical representation learning parameters
        self.eta = 1e-4  # Regularization parameter for closed-form solution

        # General intensity weights (computed via closed-form solution)
        self.W_av = None  # Will be computed during training

        # Species-specific weights (one per fish species)
        self.W_audio_species = []  # List of audio weights for each species
        self.W_visual_species = []  # List of visual weights for each species

        # Dynamic modality balancing parameters
        self.modality_weights = nn.ModuleList()  # Learnable weights for each species

        # Prototype management
        self.general_prototypes = {}  # {intensity_level: [prototypes]}
        self.species_prototypes = {}  # {fish_species: {modality: {intensity_level: [prototypes]}}}
        self.num_prototypes_per_class = 5
        self.alpha = 0.7  # Stability coefficient for prototype evolution

    def load_pretrained_backbone(self, backbone_path):
        """Load pretrained AudioVisualBackbone - much simpler!"""
        print(f"Loading pretrained backbone from: {backbone_path}")
        try:
            self.backbone = torch.load(backbone_path, map_location='cpu')
            self.backbone.eval()  # Set to evaluation mode
            print("Pretrained backbone loaded successfully!")

            # Freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

        except Exception as e:
            print(f"Error loading pretrained backbone: {e}")
            self.backbone = None

    def init_backbone_extractors(self):
        """Initialize backbone - no complex extraction needed"""
        if self.backbone is None:
            print("No pretrained backbone found!")
            # Could initialize a random backbone here if needed
            raise ValueError("Pretrained backbone is required for this method")

    def init_feature_expansion(self):
        """
        Initialize feature expansion layers as per paper:
        F^{av} = ReLU(f^{av}W_{up}^{av})
        These are randomly initialized and FIXED
        """
        # Audio-visual expansion
        self.W_up_av = nn.Linear(self.feature_dim, self.expanded_dim, bias=False)

        # Individual modality expansions
        if self.modality in ['visual', 'audio-visual']:
            self.W_up_v = nn.Linear(self.feature_dim, self.expanded_dim, bias=False)
        if self.modality in ['audio', 'audio-visual']:
            self.W_up_a = nn.Linear(self.feature_dim, self.expanded_dim, bias=False)

        # Freeze expansion weights (they remain fixed after initialization)
        for name, param in self.named_parameters():
            if 'W_up_' in name:
                param.requires_grad = False

        print(f"Feature expansion initialized: {self.feature_dim} -> {self.expanded_dim}")

    def freeze_backbone(self, freeze=True):
        """Freeze backbone - much simpler since we use the complete model"""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = not freeze
            print(f"Backbone {'frozen' if freeze else 'unfrozen'}")

    def extract_backbone_features(self, visual=None, audio=None):
        """
        Extract features using the complete pretrained backbone
        This is MUCH simpler - just use the backbone directly!
        """
        if self.backbone is None:
            raise ValueError("No pretrained backbone available")

        with torch.no_grad():  # Always no grad since backbone is frozen
            # Use the backbone's forward pass but stop before final classification
            # We need to modify this based on your backbone's structure

            # Get the fused features from backbone (before final classifier)
            # This assumes your backbone has a way to get intermediate features
            batch_size = visual.shape[0] if visual is not None else audio.shape[0]

            if self.modality == 'audio-visual':
                # Process visual dimensions to match backbone expectations
                if visual.dim() == 2:
                    visual = visual.unsqueeze(1).unsqueeze(1)
                    L, S = 1, 1
                elif visual.dim() == 3:
                    visual = visual.unsqueeze(2)
                    L, S = visual.shape[1], 1
                else:
                    L, S = visual.shape[1], visual.shape[2]

                # Project to unified dimension
                f_visual = self.backbone.visual_input_proj(visual)
                f_audio = self.backbone.audio_input_proj(audio)

                # Get attention scores
                score_audio = torch.tanh(self.backbone.W_audio(f_audio))
                f_visual_reshaped = f_visual.view(batch_size * L, S, self.feature_dim)
                score_visual_frames = torch.tanh(self.backbone.W_visual(f_visual_reshaped))
                score_visual_frames = score_visual_frames.view(batch_size, L, S, self.feature_dim)

                # Simplified attention (since we're using it for feature extraction only)
                f_prime_visual = f_visual.mean(dim=(1, 2))  # Simple pooling
                f_prime_audio = f_audio

                # Final fusion using backbone's learned weights
                enhanced_visual_proj = torch.sigmoid(self.backbone.U_visual(f_prime_visual))
                enhanced_audio_proj = torch.sigmoid(self.backbone.U_audio(f_prime_audio))
                fused_feature = enhanced_visual_proj + enhanced_audio_proj

                return fused_feature, f_prime_visual, f_prime_audio

            elif self.modality == 'visual':
                if visual.dim() == 3:
                    visual = torch.mean(visual, dim=1)
                f_visual = self.backbone.visual_input_proj(visual)
                return f_visual, f_visual, None

            elif self.modality == 'audio':
                f_audio = self.backbone.audio_input_proj(audio)
                return f_audio, None, f_audio

    def feature_expansion(self, f_av, f_v=None, f_a=None):
        """
        Feature expansion as per paper:
        F^{av} = ReLU(f^{av}W_{up}^{av})
        """
        F_av = F.relu(self.W_up_av(f_av))

        F_v = None
        F_a = None

        if f_v is not None and hasattr(self, 'W_up_v'):
            F_v = F.relu(self.W_up_v(f_v))
        if f_a is not None and hasattr(self, 'W_up_a'):
            F_a = F.relu(self.W_up_a(f_a))

        return F_av, F_v, F_a

    def compute_general_weights_closed_form(self, F_av, Y):
        """
        Compute general intensity weights using closed-form solution:
        W^{av}_k = ((F^{av}_k)^T F^{av}_k + η I)^{-1} (F^{av}_k)^T Y_k
        """
        device = F_av.device
        n_samples, n_features = F_av.shape

        # Convert labels to one-hot if needed
        if Y.dim() == 1:
            Y_onehot = F.one_hot(Y, num_classes=self.num_classes).float()
        else:
            Y_onehot = Y.float()

        # Compute closed-form solution
        F_T_F = torch.mm(F_av.T, F_av)  # F^T F
        I = torch.eye(n_features, device=device)  # Identity matrix
        regularized = F_T_F + self.eta * I  # F^T F + η I

        try:
            # Solve: (F^T F + η I)^{-1} F^T Y
            F_T_Y = torch.mm(F_av.T, Y_onehot)  # F^T Y
            W_av = torch.solve(F_T_Y, regularized)[0]  # (F^T F + η I)^{-1} F^T Y
        except:
            # Fallback to pseudo-inverse if numerical issues
            W_av = torch.pinverse(regularized) @ F_T_Y

        return W_av

    def compute_species_weights_closed_form(self, F_a, F_v, Y):
        """
        Compute species-specific weights:
        W_k^a = ((F_k^a)^T F_k^a + η I)^{-1} (F_k^a)^T Y_k
        W_k^v = ((F_k^v)^T F_k^v + η I)^{-1} (F_k^v)^T Y_k
        """
        device = F_a.device if F_a is not None else F_v.device

        # Convert labels to one-hot
        if Y.dim() == 1:
            Y_onehot = F.one_hot(Y, num_classes=self.num_classes).float()
        else:
            Y_onehot = Y.float()

        W_a, W_v = None, None

        if F_a is not None:
            n_features = F_a.shape[1]
            F_T_F = torch.mm(F_a.T, F_a)
            I = torch.eye(n_features, device=device)
            regularized = F_T_F + self.eta * I

            try:
                F_T_Y = torch.mm(F_a.T, Y_onehot)
                W_a = torch.solve(F_T_Y, regularized)[0]
            except:
                W_a = torch.pinverse(regularized) @ F_T_Y

        if F_v is not None:
            n_features = F_v.shape[1]
            F_T_F = torch.mm(F_v.T, F_v)
            I = torch.eye(n_features, device=device)
            regularized = F_T_F + self.eta * I

            try:
                F_T_Y = torch.mm(F_v.T, Y_onehot)
                W_v = torch.solve(F_T_Y, regularized)[0]
            except:
                W_v = torch.pinverse(regularized) @ F_T_Y

        return W_a, W_v

    def generate_prototypes(self, F_av, F_a, F_v, labels, fish_species_idx):
        """Generate prototypes using k-means clustering"""
        F_av_np = F_av.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Update general prototypes
        for intensity_level in range(self.num_classes):
            mask = labels_np == intensity_level
            if np.sum(mask) > 0:
                features = F_av_np[mask]

                if len(features) >= self.num_prototypes_per_class:
                    kmeans = KMeans(n_clusters=self.num_prototypes_per_class, random_state=42)
                    kmeans.fit(features)
                    new_prototypes = kmeans.cluster_centers_
                else:
                    new_prototypes = features

                # Update with exponential moving average
                if intensity_level in self.general_prototypes:
                    old_prototypes = self.general_prototypes[intensity_level]
                    updated_prototypes = (self.alpha * old_prototypes +
                                          (1 - self.alpha) * new_prototypes)
                    self.general_prototypes[intensity_level] = updated_prototypes
                else:
                    self.general_prototypes[intensity_level] = new_prototypes

        # Update species-specific prototypes
        species_key = f"fish_{fish_species_idx}"
        if species_key not in self.species_prototypes:
            self.species_prototypes[species_key] = {'audio': {}, 'visual': {}}

        # Audio prototypes
        if F_a is not None:
            F_a_np = F_a.detach().cpu().numpy()
            for intensity_level in range(self.num_classes):
                mask = labels_np == intensity_level
                if np.sum(mask) > 0:
                    features = F_a_np[mask]
                    if len(features) >= self.num_prototypes_per_class:
                        kmeans = KMeans(n_clusters=self.num_prototypes_per_class, random_state=42)
                        kmeans.fit(features)
                        prototypes = kmeans.cluster_centers_
                    else:
                        prototypes = features
                    self.species_prototypes[species_key]['audio'][intensity_level] = prototypes

        # Visual prototypes
        if F_v is not None:
            F_v_np = F_v.detach().cpu().numpy()
            for intensity_level in range(self.num_classes):
                mask = labels_np == intensity_level
                if np.sum(mask) > 0:
                    features = F_v_np[mask]
                    if len(features) >= self.num_prototypes_per_class:
                        kmeans = KMeans(n_clusters=self.num_prototypes_per_class, random_state=42)
                        kmeans.fit(features)
                        prototypes = kmeans.cluster_centers_
                    else:
                        prototypes = features
                    self.species_prototypes[species_key]['visual'][intensity_level] = prototypes

    def prototype_enhanced_training(self, F_av, Y, fish_species_idx):
        """
        Prototype-enhanced training as per paper:
        F̃_k^{av} = [F_k^{av}; λ_p · P^{av}]
        """
        if not self.general_prototypes:
            return F_av, Y

        device = F_av.device
        prototype_features = []
        prototype_labels = []

        for intensity_level, prototypes in self.general_prototypes.items():
            prototype_features.append(torch.tensor(prototypes, dtype=torch.float32, device=device))
            intensity_labels = torch.full((len(prototypes),), intensity_level, dtype=torch.long, device=device)
            prototype_labels.append(intensity_labels)

        if prototype_features:
            P_av = torch.cat(prototype_features, dim=0)  # [num_prototypes, expanded_dim]
            I_proto = torch.cat(prototype_labels, dim=0)  # [num_prototypes]

            # Compute similarity-based weighting λ_p
            similarities = F.cosine_similarity(F_av.unsqueeze(1), P_av.unsqueeze(0), dim=2)
            avg_similarity = similarities.mean().item()
            lambda_p = max(0.2, avg_similarity)  # Minimum threshold as per paper

            # Augment features
            weighted_prototypes = lambda_p * P_av
            F_av_tilde = torch.cat([F_av, weighted_prototypes], dim=0)

            # Augment labels
            if Y.dim() == 1:
                Y_tilde = torch.cat([Y, I_proto], dim=0)
            else:
                I_proto_onehot = F.one_hot(I_proto, num_classes=self.num_classes).float()
                Y_tilde = torch.cat([Y, I_proto_onehot], dim=0)

            return F_av_tilde, Y_tilde

        return F_av, Y

    def add_species_modality_balancing(self, fish_species_idx):
        """Add modality balancing parameters for new species"""
        if self.modality == 'audio-visual':
            # Learnable weights for dynamic modality balancing
            modality_net = nn.Sequential(
                nn.Linear(self.expanded_dim * 2, 64),  # Concatenated F_a and F_v
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            self.modality_weights.append(modality_net)

    def train_incremental_step(self, visual, audio, labels, fish_species_idx):
        """Train one incremental step following the paper methodology"""
        # 1. Extract backbone features using the complete pretrained model
        f_av, f_v, f_a = self.extract_backbone_features(visual, audio)

        # 2. Feature expansion
        F_av, F_v, F_a = self.feature_expansion(f_av, f_v, f_a)

        # 3. Prototype-enhanced training
        F_av_tilde, Y_tilde = self.prototype_enhanced_training(F_av, labels, fish_species_idx)

        # 4. Compute general weights using closed-form solution
        self.W_av = self.compute_general_weights_closed_form(F_av_tilde, Y_tilde)

        # 5. Compute species-specific weights
        W_a, W_v = self.compute_species_weights_closed_form(F_a, F_v, labels)

        # Store species weights
        if len(self.W_audio_species) <= fish_species_idx:
            self.W_audio_species.append(W_a)
            self.W_visual_species.append(W_v)
            self.add_species_modality_balancing(fish_species_idx)
        else:
            self.W_audio_species[fish_species_idx] = W_a
            self.W_visual_species[fish_species_idx] = W_v

        # 6. Generate/update prototypes
        self.generate_prototypes(F_av, F_a, F_v, labels, fish_species_idx)

        print(f"Trained incremental step {fish_species_idx}")

    def forward(self, visual=None, audio=None, fish_species_idx=None):
        """
        Forward pass with dynamic modality balancing as per paper:
        ŷ = softmax(γ_k · ŷ_k^{av} + (1-γ_k) · ŷ_k^{sp})
        """
        # Extract and expand features using the pretrained backbone
        f_av, f_v, f_a = self.extract_backbone_features(visual, audio)
        F_av, F_v, F_a = self.feature_expansion(f_av, f_v, f_a)

        # General intensity prediction
        if self.W_av is not None:
            y_av = torch.mm(F_av, self.W_av)  # (W^{av}_k)^T F^{av}_k
        else:
            y_av = torch.zeros(F_av.shape[0], self.num_classes, device=F_av.device)

        # Species-specific prediction
        if (fish_species_idx is not None and
                fish_species_idx < len(self.W_audio_species) and
                self.modality == 'audio-visual'):

            W_a = self.W_audio_species[fish_species_idx]
            W_v = self.W_visual_species[fish_species_idx]

            if W_a is not None and W_v is not None:
                # Compute modality weights
                concat_features = torch.cat([F_a, F_v], dim=1)
                beta_a = self.modality_weights[fish_species_idx](concat_features)
                beta_v = 1 - beta_a

                y_a = torch.mm(F_a, W_a)
                y_v = torch.mm(F_v, W_v)
                y_sp = beta_a * y_a + beta_v * y_v
            else:
                y_sp = torch.zeros_like(y_av)
        else:
            y_sp = torch.zeros_like(y_av)

        # Dynamic balancing parameter γ_k
        total_species = 6  # As mentioned in paper
        if fish_species_idx is not None:
            gamma_max, gamma_min = 0.8, 0.3
            gamma_k = gamma_max - (gamma_max - gamma_min) * (fish_species_idx / total_species)
        else:
            gamma_k = 0.8

        # Final prediction
        final_logits = gamma_k * y_av + (1 - gamma_k) * y_sp

        return F.softmax(final_logits, dim=1)

    def set_incremental_step(self, step):
        """Set current incremental step"""
        self.incremental_step = step


# Wrapper for compatibility
class IncreAudioVisualNet(HAIL_FFIR_Net):
    def __init__(self, args, step_out_class_num, LSC=False, pretrained_backbone_path=None):
        super(IncreAudioVisualNet, self).__init__(args, step_out_class_num, 0, pretrained_backbone_path)

    def incremental_classifier(self, numclass):
        pass

    def forward(self, visual=None, audio=None, out_logits=True, fish_species_idx=None, labels=None):
        if self.training and labels is not None and fish_species_idx is not None:
            # Training mode: use the incremental training method
            self.train_incremental_step(visual, audio, labels, fish_species_idx)

        # Always return predictions
        return super().forward(visual, audio, fish_species_idx)


def create_model_with_pretrained_backbone(args, step_out_class_num, backbone_path, freeze_backbone=True):
    """Create model with pretrained backbone following paper methodology"""
    model = IncreAudioVisualNet(args, step_out_class_num, pretrained_backbone_path=backbone_path)

    if freeze_backbone:
        model.freeze_backbone(freeze=True)
        print("Backbone frozen - using paper's closed-form learning")

    return model


if __name__ == "__main__":
    # Test the implementation
    class Args:
        def __init__(self):
            self.modality = 'audio-visual'


    args = Args()
    model = create_model_with_pretrained_backbone(
        args, 4, './save/RedTilapia/audio_visual/best_backbone_model.pkl'
    )

    print("HAIL-FFIR model created following paper methodology!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    visual = torch.randn(8, 10, 1028)
    audio = torch.randn(8, 1024)
    labels = torch.randint(0, 4, (8,))

    with torch.no_grad():
        output = model(visual, audio, fish_species_idx=0, labels=labels)
        print(f"Output shape: {output.shape}")
        print("Model ready for incremental learning!")