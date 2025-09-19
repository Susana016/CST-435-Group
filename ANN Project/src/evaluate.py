"""
Evaluation Module for NBA Team Selection Model
Handles model evaluation, metrics calculation, and visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import pandas as pd

class Evaluator:
    """
    Evaluator class for comprehensive model assessment.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            device: Device for evaluation
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        all_outputs = []
        all_targets = []
        all_names = []
        
        with torch.no_grad():
            for features, targets, names in test_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                
                all_outputs.append(outputs.cpu())
                all_targets.append(targets)
                if names[0] is not None:
                    all_names.extend(names)
        
        # Concatenate all batches
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Evaluate position classification
        position_metrics = self._evaluate_positions(all_outputs, all_targets)
        
        # Evaluate team fit scores
        team_fit_metrics = self._evaluate_team_fit(all_outputs, all_targets)
        
        # Combine metrics
        metrics = {
            'position_metrics': position_metrics,
            'team_fit_metrics': team_fit_metrics,
            'player_evaluations': self._create_player_evaluations(
                all_outputs, all_targets, all_names
            )
        }
        
        return metrics
    
    def _evaluate_positions(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        Evaluate position classification performance.
        
        Args:
            outputs: Model outputs
            targets: True targets
            
        Returns:
            Position classification metrics
        """
        # Get position predictions
        position_probs = self.model.get_position_predictions(outputs)
        position_preds = position_probs.argmax(dim=1).numpy()
        position_true = targets[:, :3].argmax(dim=1).numpy()
        
        # Calculate metrics
        accuracy = (position_preds == position_true).mean()
        
        # Confusion matrix - ensure 3x3 even if not all classes are present
        cm = confusion_matrix(position_true, position_preds, labels=[0, 1, 2])
        
        # Classification report
        position_names = ['Guard', 'Forward', 'Center']
        # Ensure we have labels 0, 1, 2 even if not all are in the test set
        labels = [0, 1, 2]
        report = classification_report(
            position_true, position_preds,
            target_names=position_names,
            labels=labels,
            zero_division=0,
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': position_preds,
            'true_labels': position_true,
            'probabilities': position_probs.numpy()
        }
    
    def _evaluate_team_fit(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        Evaluate team fit score regression.
        
        Args:
            outputs: Model outputs
            targets: True targets
            
        Returns:
            Team fit regression metrics
        """
        # Get team fit scores
        team_fit_preds = self.model.get_team_fit_score(outputs).numpy()
        team_fit_true = targets[:, -1].numpy()
        
        # Calculate metrics
        mse = mean_squared_error(team_fit_true, team_fit_preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(team_fit_true, team_fit_preds)
        mae = np.mean(np.abs(team_fit_true - team_fit_preds))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mae': mae,
            'predictions': team_fit_preds,
            'true_scores': team_fit_true
        }
    
    def _create_player_evaluations(self, 
                                  outputs: torch.Tensor,
                                  targets: torch.Tensor,
                                  names: List[str]) -> pd.DataFrame:
        """
        Create detailed player evaluations.
        
        Args:
            outputs: Model outputs
            targets: True targets
            names: Player names
            
        Returns:
            DataFrame with player evaluations
        """
        position_probs = self.model.get_position_predictions(outputs).numpy()
        position_preds = position_probs.argmax(axis=1)
        team_fit_scores = self.model.get_team_fit_score(outputs).numpy()
        
        position_names = ['Guard', 'Forward', 'Center']
        
        evaluations = []
        for i in range(len(outputs)):
            eval_dict = {
                'player_name': names[i] if names else f'Player_{i}',
                'predicted_position': position_names[position_preds[i]],
                'guard_prob': position_probs[i, 0],
                'forward_prob': position_probs[i, 1],
                'center_prob': position_probs[i, 2],
                'team_fit_score': team_fit_scores[i],
                'true_team_fit': targets[i, -1].item()
            }
            evaluations.append(eval_dict)
        
        return pd.DataFrame(evaluations)

def plot_training_history(history: Dict, save_path: str = 'outputs/training_history.png'):
    """
    Plot training history curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[1, 0].plot(history['learning_rates'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Loss difference (overfitting indicator)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 1].plot(loss_diff, linewidth=2, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Loss - Train Loss')
    axes[1, 1].set_title('Overfitting Indicator')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to {save_path}")

def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str] = ['Guard', 'Forward', 'Center'],
                         save_path: str = 'outputs/confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    
    plt.xlabel('Predicted Position', fontsize=12)
    plt.ylabel('True Position', fontsize=12)
    plt.title('Position Classification Confusion Matrix', fontsize=14)
    
    # Add counts as text
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f'n={cm[i,j]}',
                    ha='center', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")

def plot_team_fit_predictions(predictions: np.ndarray,
                              true_scores: np.ndarray,
                              save_path: str = 'outputs/team_fit_scatter.png'):
    """
    Plot team fit score predictions vs true scores.
    
    Args:
        predictions: Predicted team fit scores
        true_scores: True team fit scores
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    plt.scatter(true_scores, predictions, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(true_scores.min(), predictions.min())
    max_val = max(true_scores.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R2
    r2 = r2_score(true_scores, predictions)
    
    plt.xlabel('True Team Fit Score', fontsize=12)
    plt.ylabel('Predicted Team Fit Score', fontsize=12)
    plt.title(f'Team Fit Score Predictions (R² = {r2:.3f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Team fit scatter plot saved to {save_path}")

def generate_evaluation_report(metrics: Dict, save_path: str = 'outputs/evaluation_report.txt'):
    """
    Generate comprehensive evaluation report.
    
    Args:
        metrics: Evaluation metrics dictionary
        save_path: Path to save the report
    """
    report = []
    report.append("=" * 60)
    report.append("NBA TEAM SELECTION MODEL - EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Position Classification Metrics
    report.append("POSITION CLASSIFICATION METRICS:")
    report.append("-" * 40)
    pos_metrics = metrics['position_metrics']
    report.append(f"Overall Accuracy: {pos_metrics['accuracy']:.4f}")
    report.append("")
    report.append("Per-Class Metrics:")
    
    for class_name in ['Guard', 'Forward', 'Center']:
        if class_name in pos_metrics['classification_report']:
            class_metrics = pos_metrics['classification_report'][class_name]
            report.append(f"  {class_name}:")
            report.append(f"    Precision: {class_metrics['precision']:.4f}")
            report.append(f"    Recall: {class_metrics['recall']:.4f}")
            report.append(f"    F1-Score: {class_metrics['f1-score']:.4f}")
        else:
            report.append(f"  {class_name}: No samples in test set")
    
    report.append("")
    
    # Team Fit Regression Metrics
    report.append("TEAM FIT SCORE METRICS:")
    report.append("-" * 40)
    fit_metrics = metrics['team_fit_metrics']
    report.append(f"Mean Squared Error (MSE): {fit_metrics['mse']:.6f}")
    report.append(f"Root Mean Squared Error (RMSE): {fit_metrics['rmse']:.6f}")
    report.append(f"Mean Absolute Error (MAE): {fit_metrics['mae']:.6f}")
    report.append(f"R² Score: {fit_metrics['r2_score']:.4f}")
    
    report.append("")
    report.append("=" * 60)
    
    # Save report
    report_text = "\n".join(report)
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nEvaluation report saved to {save_path}")