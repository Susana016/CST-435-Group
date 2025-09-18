"""
Utility Functions for NBA Team Selection Project
Helper functions for model management, visualization, and data processing
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pickle

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'outputs', 'models', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directories created/verified")

def save_model(model: nn.Module, 
               optimizer: torch.optim.Optimizer,
               epoch: int,
               metrics: Dict,
               filepath: str = 'models/nba_model.pth'):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Performance metrics
        filepath: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(model: nn.Module,
               filepath: str = 'models/nba_model.pth',
               optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model architecture
        filepath: Path to checkpoint
        optimizer: Optimizer to load state (optional)
        
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    return checkpoint

def save_preprocessor(preprocessor: Any, filepath: str = 'models/preprocessor.pkl'):
    """
    Save preprocessor object.
    
    Args:
        preprocessor: Preprocessor object
        filepath: Save path
    """
    with open(filepath, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {filepath}")

def load_preprocessor(filepath: str = 'models/preprocessor.pkl') -> Any:
    """
    Load preprocessor object.
    
    Args:
        filepath: Path to preprocessor file
        
    Returns:
        Preprocessor object
    """
    with open(filepath, 'rb') as f:
        preprocessor = pickle.load(f)
    print(f"Preprocessor loaded from {filepath}")
    return preprocessor

def save_config(config: Dict, filepath: str = 'config.json'):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Save path
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {filepath}")

def load_config(filepath: str = 'config.json') -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {filepath}")
    return config

def plot_player_distribution(df: pd.DataFrame, save_path: str = 'outputs/player_distribution.png'):
    """
    Plot distribution of player statistics.
    
    Args:
        df: Player data DataFrame
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Points distribution
    axes[0, 0].hist(df['pts'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Points per Game')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Points Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rebounds distribution
    axes[0, 1].hist(df['reb'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Rebounds per Game')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Rebounds Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Assists distribution
    axes[0, 2].hist(df['ast'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 2].set_xlabel('Assists per Game')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Assists Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Height vs Weight
    axes[1, 0].scatter(df['player_height'], df['player_weight'], alpha=0.6)
    axes[1, 0].set_xlabel('Height (cm)')
    axes[1, 0].set_ylabel('Weight (kg)')
    axes[1, 0].set_title('Height vs Weight')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Age distribution
    axes[1, 1].hist(df['age'], bins=20, edgecolor='black', alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Age')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Age Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Net rating distribution
    axes[1, 2].hist(df['net_rating'], bins=30, edgecolor='black', alpha=0.7, color='red')
    axes[1, 2].set_xlabel('Net Rating')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Net Rating Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Player distribution plot saved to {save_path}")

def plot_correlation_matrix(df: pd.DataFrame, 
                           features: List[str],
                           save_path: str = 'outputs/correlation_matrix.png'):
    """
    Plot correlation matrix of features.
    
    Args:
        df: DataFrame with features
        features: List of feature columns
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5)
    
    plt.title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Correlation matrix saved to {save_path}")

def create_model_summary(model: nn.Module) -> pd.DataFrame:
    """
    Create a summary of model architecture.
    
    Args:
        model: PyTorch model
        
    Returns:
        DataFrame with layer information
    """
    summary_data = []
    total_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            
            summary_data.append({
                'Layer': name,
                'Type': module.__class__.__name__,
                'Parameters': params,
                'Output Shape': str(getattr(module, 'out_features', 'N/A'))
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.loc[len(summary_df)] = ['Total', '-', total_params, '-']
    
    return summary_df

def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")

def get_device() -> str:
    """
    Get the best available device.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device

def create_experiment_log(experiment_name: str,
                         config: Dict,
                         metrics: Dict,
                         save_path: str = 'logs/experiments.json'):
    """
    Log experiment details.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        metrics: Experiment results
        save_path: Path to save log
    """
    log_entry = {
        'name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'metrics': metrics
    }
    
    # Load existing logs if file exists
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    # Save updated logs
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(logs, f, indent=4)
    
    print(f"Experiment logged: {experiment_name}")

def print_banner(text: str, width: int = 60):
    """
    Print a formatted banner.
    
    Args:
        text: Text to display
        width: Banner width
    """
    print("=" * width)
    print(text.center(width))
    print("=" * width)