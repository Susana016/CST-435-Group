"""
PyTorch Dataset and DataLoader for NBA Players
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional

class NBAPlayersDataset(Dataset):
    """
    Custom PyTorch Dataset for NBA Players data.
    """
    
    def __init__(self, 
                 features: np.ndarray,
                 targets: Optional[np.ndarray] = None,
                 player_names: Optional[list] = None,
                 transform=None):
        """
        Initialize the dataset.
        
        Args:
            features: Input features array
            targets: Target values array (optional for inference)
            player_names: List of player names for reference
            transform: Optional transform to apply to samples
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        self.player_names = player_names
        self.transform = transform
        
        # Validate dimensions
        if self.targets is not None:
            assert len(self.features) == len(self.targets), \
                "Features and targets must have same length"
        
        print(f"Dataset initialized with {len(self.features)} samples")
        print(f"Feature shape: {self.features.shape}")
        if self.targets is not None:
            print(f"Target shape: {self.targets.shape}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[str]]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, targets, player_name)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_features = self.features[idx]
        
        # Apply transform if available
        if self.transform:
            sample_features = self.transform(sample_features)
        
        sample_targets = self.targets[idx] if self.targets is not None else None
        player_name = self.player_names[idx] if self.player_names is not None else None
        
        return sample_features, sample_targets, player_name
    
    def get_feature_dim(self) -> int:
        """Get the dimension of input features."""
        return self.features.shape[1]
    
    def get_target_dim(self) -> int:
        """Get the dimension of targets."""
        if self.targets is not None:
            return self.targets.shape[1] if len(self.targets.shape) > 1 else 1
        return 0

def create_data_loaders(train_features: np.ndarray,
                       train_targets: np.ndarray,
                       val_features: np.ndarray,
                       val_targets: np.ndarray,
                       test_features: np.ndarray,
                       test_targets: np.ndarray,
                       train_names: Optional[list] = None,
                       val_names: Optional[list] = None,
                       test_names: Optional[list] = None,
                       batch_size: int = 16,
                       shuffle_train: bool = True,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        train_features: Training features
        train_targets: Training targets
        val_features: Validation features
        val_targets: Validation targets
        test_features: Test features
        test_targets: Test targets
        train_names: Training player names
        val_names: Validation player names
        test_names: Test player names
        batch_size: Batch size for DataLoader
        shuffle_train: Whether to shuffle training data
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = NBAPlayersDataset(train_features, train_targets, train_names)
    val_dataset = NBAPlayersDataset(val_features, val_targets, val_names)
    test_dataset = NBAPlayersDataset(test_features, test_targets, test_names)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"DataLoaders created with batch_size={batch_size}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

class DataAugmentation:
    """
    Data augmentation techniques for NBA player features.
    """
    
    def __init__(self, noise_std: float = 0.01, dropout_prob: float = 0.1):
        """
        Initialize data augmentation.
        
        Args:
            noise_std: Standard deviation for Gaussian noise
            dropout_prob: Probability of dropping features
        """
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
    
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to features.
        
        Args:
            features: Input features tensor
            
        Returns:
            Augmented features tensor
        """
        augmented = features.clone()
        
        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(features) * self.noise_std
            augmented = augmented + noise
        
        # Random feature dropout
        if self.dropout_prob > 0:
            mask = torch.rand_like(features) > self.dropout_prob
            augmented = augmented * mask.float()
        
        return augmented