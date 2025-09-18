"""
Artificial Neural Network (MLP) Model for NBA Team Selection
This module defines the deep MLP architecture with multiple hidden layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class NBATeamMLP(nn.Module):
    """
    Multi-Layer Perceptron for NBA Team Selection.
    
    Architecture:
    - Input Layer: Receives player features (stats, physical attributes)
    - Hidden Layers: Multiple fully connected layers with activation functions
    - Output Layer: Produces team fit scores and position predictions
    
    The network learns to:
    1. Identify player positions based on statistics
    2. Evaluate team fit scores for optimal team composition
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32, 16],
                 output_dim: int = 4,  # 3 for position + 1 for team fit score
                 dropout_rate: float = 0.2,
                 activation: str = 'relu',
                 use_batch_norm: bool = True):
        """
        Initialize the MLP architecture.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output neurons
            dropout_rate: Dropout probability for regularization
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'tanh')
            use_batch_norm: Whether to use batch normalization
        """
        super(NBATeamMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Select activation function
        self.activation_func = self._get_activation(activation)
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self.activation_func)
            
            # Dropout (except for the last hidden layer)
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Create the sequential model for hidden layers
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer (no activation - will apply task-specific activation later)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"MLP Architecture created:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Hidden layers: {hidden_dims}")
        print(f"  Output dimension: {output_dim}")
        print(f"  Total parameters: {self.count_parameters():,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Pass through hidden layers
        hidden = self.hidden_layers(x)
        
        # Pass through output layer
        output = self.output_layer(hidden)
        
        return output
    
    def forward_with_intermediates(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that also returns intermediate activations.
        Useful for visualization and analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (output, intermediate_activations)
        """
        intermediates = []
        current = x
        
        for layer in self.hidden_layers:
            current = layer(current)
            if isinstance(layer, nn.Linear):
                intermediates.append(current.detach())
        
        output = self.output_layer(current)
        intermediates.append(output.detach())
        
        return output, intermediates
    
    def _get_activation(self, activation: str):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU activation
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_position_predictions(self, output: torch.Tensor) -> torch.Tensor:
        """
        Extract position predictions from network output.
        
        Args:
            output: Network output tensor
            
        Returns:
            Position predictions (first 3 outputs with softmax)
        """
        # Apply softmax to first 3 outputs (position classification)
        position_logits = output[:, :3]
        position_probs = F.softmax(position_logits, dim=1)
        return position_probs
    
    def get_team_fit_score(self, output: torch.Tensor) -> torch.Tensor:
        """
        Extract team fit score from network output.
        
        Args:
            output: Network output tensor
            
        Returns:
            Team fit scores (last output with sigmoid)
        """
        # Apply sigmoid to last output (team fit score)
        team_fit_logit = output[:, -1]
        team_fit_score = torch.sigmoid(team_fit_logit)
        return team_fit_score

class CustomLoss(nn.Module):
    """
    Custom loss function for NBA team selection.
    Combines position classification loss and team fit regression loss.
    """
    
    def __init__(self, 
                 position_weight: float = 0.6,
                 team_fit_weight: float = 0.4):
        """
        Initialize custom loss.
        
        Args:
            position_weight: Weight for position classification loss
            team_fit_weight: Weight for team fit score loss
        """
        super(CustomLoss, self).__init__()
        self.position_weight = position_weight
        self.team_fit_weight = team_fit_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            output: Network output (batch_size, 4)
            target: Target values (batch_size, 4)
            
        Returns:
            Combined loss value
        """
        # Position classification loss (first 3 outputs)
        position_output = output[:, :3]
        position_target = target[:, :3].argmax(dim=1)  # Convert one-hot to class indices
        position_loss = self.ce_loss(position_output, position_target)
        
        # Team fit score loss (last output)
        team_fit_output = torch.sigmoid(output[:, -1])
        team_fit_target = target[:, -1]
        team_fit_loss = self.mse_loss(team_fit_output, team_fit_target)
        
        # Combine losses
        total_loss = self.position_weight * position_loss + self.team_fit_weight * team_fit_loss
        
        return total_loss

def create_model(input_dim: int,
                 config: Optional[dict] = None) -> NBATeamMLP:
    """
    Factory function to create model with configuration.
    
    Args:
        input_dim: Number of input features
        config: Configuration dictionary
        
    Returns:
        Initialized NBATeamMLP model
    """
    if config is None:
        config = {
            'hidden_dims': [128, 64, 32, 16],
            'output_dim': 4,
            'dropout_rate': 0.2,
            'activation': 'relu',
            'use_batch_norm': True
        }
    
    model = NBATeamMLP(
        input_dim=input_dim,
        hidden_dims=config.get('hidden_dims', [128, 64, 32, 16]),
        output_dim=config.get('output_dim', 4),
        dropout_rate=config.get('dropout_rate', 0.2),
        activation=config.get('activation', 'relu'),
        use_batch_norm=config.get('use_batch_norm', True)
    )
    
    return model