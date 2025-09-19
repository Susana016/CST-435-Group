"""
Training Module for NBA Team Selection MLP
Implements the training loop with forward propagation, loss calculation,
backpropagation, and weight updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

class Trainer:
    """
    Trainer class for the NBA Team Selection MLP.
    Handles the complete training process including:
    - Forward propagation
    - Loss calculation
    - Backpropagation
    - Weight updates via optimizer
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 optimizer_type: str = 'adam',
                 scheduler_type: str = 'step',
                 patience: int = 10):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model to train
            device: Device to run training on ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            scheduler_type: Type of learning rate scheduler ('step', 'plateau', 'none')
            patience: Patience for early stopping
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer_type)
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler(scheduler_type)
        
        # Loss functions
        self.criterion = CustomLoss(position_weight=0.6, team_fit_weight=0.4)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        print(f"Trainer initialized on {device}")
        print(f"Optimizer: {optimizer_type}, LR: {learning_rate}")
        print(f"Scheduler: {scheduler_type}")
    
    def _create_optimizer(self, optimizer_type: str) -> optim.Optimizer:
        """Create optimizer based on type."""
        optimizers = {
            'adam': optim.Adam(self.model.parameters(), 
                              lr=self.learning_rate, 
                              weight_decay=self.weight_decay),
            'sgd': optim.SGD(self.model.parameters(), 
                            lr=self.learning_rate, 
                            momentum=0.9,
                            weight_decay=self.weight_decay),
            'rmsprop': optim.RMSprop(self.model.parameters(), 
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        }
        return optimizers.get(optimizer_type.lower(), optimizers['adam'])
    
    def _create_scheduler(self, scheduler_type: str):
        """Create learning rate scheduler."""
        if scheduler_type == 'step':
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(self.optimizer, mode='min', 
                                    factor=0.5, patience=5)
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (features, targets, _) in enumerate(progress_bar):
            # Move to device
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients from previous step
            self.optimizer.zero_grad()
            
            # Forward propagation
            outputs = self.model(features)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backpropagation - compute gradients
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights using optimizer
            self.optimizer.step()
            
            # Calculate accuracy for position predictions
            position_preds = self.model.get_position_predictions(outputs)
            position_targets = targets[:, :3].argmax(dim=1)
            position_predicted = position_preds.argmax(dim=1)
            correct_predictions += (position_predicted == position_targets).sum().item()
            
            # Update metrics
            total_loss += loss.item()
            total_samples += features.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct_predictions / total_samples
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct_predictions / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, targets, _ in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Calculate accuracy
                position_preds = self.model.get_position_predictions(outputs)
                position_targets = targets[:, :3].argmax(dim=1)
                position_predicted = position_preds.argmax(dim=1)
                correct_predictions += (position_predicted == position_targets).sum().item()
                
                total_loss += loss.item()
                total_samples += features.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = correct_predictions / total_samples
        
        return avg_loss, avg_acc
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              save_best: bool = True,
              save_path: str = 'best_model.pth') -> Dict:
        """
        Complete training loop for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            save_best: Whether to save the best model
            save_path: Path to save the best model
            
        Returns:
            Training history dictionary
        """
        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 50)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                
                if save_best:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }, save_path)
            else:
                self.patience_counter += 1
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch:3d}/{epochs}] - {epoch_time:.2f}s - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Load best model weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model with val_loss: {self.best_val_loss:.4f}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print("=" * 50)
        
        return self.history

class CustomLoss(nn.Module):
    """Custom loss function combining position and team fit losses."""
    
    def __init__(self, position_weight: float = 0.6, team_fit_weight: float = 0.4):
        super(CustomLoss, self).__init__()
        self.position_weight = position_weight
        self.team_fit_weight = team_fit_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Position classification loss
        position_output = output[:, :3]
        position_target = target[:, :3].argmax(dim=1)
        position_loss = self.ce_loss(position_output, position_target)
        
        # Team fit score loss
        team_fit_output = torch.sigmoid(output[:, -1])
        team_fit_target = target[:, -1]
        team_fit_loss = self.mse_loss(team_fit_output, team_fit_target)
        
        # Combined loss
        total_loss = self.position_weight * position_loss + self.team_fit_weight * team_fit_loss
        
        return total_loss