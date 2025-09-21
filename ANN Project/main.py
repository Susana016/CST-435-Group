"""
Main Pipeline Script for NBA Team Selection Project
This script runs the complete pipeline from data loading to team selection
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Import project modules
from src.load_data import load_nba_data, get_feature_columns, create_position_labels, split_data
from src.preprocess import NBADataPreprocessor, create_feature_tensors, normalize_targets
from src.dataset import create_data_loaders
from src.model import create_model
from src.train import Trainer
from src.evaluate import (
    Evaluator, 
    plot_training_history, 
    plot_confusion_matrix, 
    plot_team_fit_predictions,
    generate_evaluation_report
)
from src.select_team import TeamSelector, compare_selection_methods, save_team_selection
from src.utils import (
    ensure_directories, 
    save_model, 
    save_preprocessor,
    save_config,
    set_random_seeds,
    get_device,
    plot_player_distribution,
    plot_correlation_matrix,
    create_experiment_log,
    print_banner
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NBA Team Selection using ANN')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/nba_players.csv',
                       help='Path to NBA players dataset')
    parser.add_argument('--start_year', type=str, default='2015-16',
                       help='Start year for data filtering')
    parser.add_argument('--end_year', type=str, default='2019-20',
                       help='End year for data filtering')
    parser.add_argument('--n_players', type=int, default=100,
                       help='Number of players to select')
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=str, default='128,64,32,16',
                       help='Hidden layer dimensions (comma-separated)')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout rate for regularization')
    parser.add_argument('--batch_norm', action='store_true', default=True,
                       help='Use batch normalization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                       help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                       help='Weight decay for regularization')
    
    # Team selection arguments
    parser.add_argument('--selection_method', type=str, default='balanced',
                       choices=['greedy', 'balanced', 'exhaustive'],
                       help='Team selection method')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for computation')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save visualization plots')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed logs')
    
    return parser.parse_args()

def main():
    """Main pipeline execution."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup
    print_banner("NBA TEAM SELECTION USING ARTIFICIAL NEURAL NETWORKS")
    ensure_directories()
    set_random_seeds(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    
    # Save configuration
    config = vars(args)
    save_config(config, 'config.json')
    
    # ==========================================
    # 1. DATA LOADING AND PREPARATION
    # ==========================================
    print_banner("STEP 1: DATA LOADING")
    
    # Load data
    df = load_nba_data(
        args.data_path,
        start_year=args.start_year,
        end_year=args.end_year,
        n_players=args.n_players,
        random_state=args.seed
    )
    
    # Create position labels
    df = create_position_labels(df)
    
    # Get feature columns
    numerical_features, categorical_features, target_features = get_feature_columns()
    
    # Split data
    train_df, val_df, test_df = split_data(df, random_state=args.seed)
    
    print(f"Data loaded: {len(df)} total players")
    print(f"Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")
    
    # ==========================================
    # 2. DATA PREPROCESSING
    # ==========================================
    print_banner("STEP 2: DATA PREPROCESSING")
    
    # Initialize preprocessor
    preprocessor = NBADataPreprocessor(scaler_type='standard', handle_missing='mean')
    
    # Preprocess features
    train_features, _ = preprocessor.fit_transform(train_df, numerical_features, categorical_features)
    val_features, _ = preprocessor.transform(val_df, numerical_features, categorical_features)
    test_features, _ = preprocessor.transform(test_df, numerical_features, categorical_features)
    
    # Prepare targets (position one-hot encoding + team fit score)
    def prepare_targets(df):
        position_scores = df[['guard_score', 'forward_score', 'center_score']].values
        # Convert to one-hot encoding
        position_onehot = (position_scores == position_scores.max(axis=1, keepdims=True)).astype(float)
        team_fit = df['team_fit_score'].values.reshape(-1, 1)
        return np.hstack([position_onehot, team_fit])
    
    train_targets = prepare_targets(train_df)
    val_targets = prepare_targets(val_df)
    test_targets = prepare_targets(test_df)
    
    # Save preprocessor
    save_preprocessor(preprocessor, 'models/preprocessor.pkl')
    
    print(f"Features shape: {train_features.shape}")
    print(f"Targets shape: {train_targets.shape}")
    
    # ==========================================
    # 3. CREATE DATA LOADERS
    # ==========================================
    print_banner("STEP 3: CREATING DATA LOADERS")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_features, train_targets,
        val_features, val_targets,
        test_features, test_targets,
        train_names=train_df['player_name'].tolist(),
        val_names=val_df['player_name'].tolist(),
        test_names=test_df['player_name'].tolist(),
        batch_size=args.batch_size,
        shuffle_train=True
    )
    
    # ==========================================
    # 4. BUILD MODEL
    # ==========================================
    print_banner("STEP 4: BUILDING NEURAL NETWORK")
    
    # Parse hidden dimensions
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    # Create model
    model_config = {
        'hidden_dims': hidden_dims,
        'output_dim': 4,  # 3 positions + 1 team fit
        'dropout_rate': args.dropout_rate,
        'activation': 'relu',
        'use_batch_norm': args.batch_norm
    }
    
    model = create_model(
        input_dim=train_features.shape[1],
        config=model_config
    )
    
    # ==========================================
    # 5. TRAIN MODEL
    # ==========================================
    print_banner("STEP 5: TRAINING NEURAL NETWORK")
    
    # Compute class weights for handling imbalance (device not needed, handled by CustomLoss)
    from src.utils import compute_class_weights
    class_weights = compute_class_weights(train_targets)
    
    # Create trainer with class weights
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay * 2,  # Moderate weight decay
        optimizer_type='adam',
        scheduler_type='plateau',
        patience=50,  # Much more patience
        class_weights=class_weights
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_best=True,
        save_path='models/best_model.pth'
    )
    
    # Save final model
    save_model(model, trainer.optimizer, args.epochs, history, 'models/final_model.pth')
    
    # Plot training history
    if args.save_plots:
        plot_training_history(history, 'outputs/training_history.png')
    
    # ==========================================
    # 6. EVALUATE MODEL
    # ==========================================
    print_banner("STEP 6: MODEL EVALUATION")
    
    # Create evaluator
    evaluator = Evaluator(model, device)
    
    # Evaluate on test set
    metrics = evaluator.evaluate(test_loader)
    
    # Generate evaluation report
    generate_evaluation_report(metrics, 'outputs/evaluation_report.txt')
    
    # Plot evaluation visualizations
    if args.save_plots:
        plot_confusion_matrix(
            metrics['position_metrics']['confusion_matrix'],
            save_path='outputs/confusion_matrix.png'
        )
        plot_team_fit_predictions(
            metrics['team_fit_metrics']['predictions'],
            metrics['team_fit_metrics']['true_scores'],
            save_path='outputs/team_fit_scatter.png'
        )
    
    # ==========================================
    # 7. SELECT OPTIMAL TEAM
    # ==========================================
    print_banner("STEP 7: OPTIMAL TEAM SELECTION")
    
    # Prepare all data for team selection
    all_features, _ = preprocessor.transform(df, numerical_features, categorical_features)
    
    # Create team selector
    selector = TeamSelector(model, device)
    
    # Evaluate all players
    evaluations = selector.evaluate_players(
        features=all_features,
        player_names=df['player_name'].tolist(),
        player_stats=df
    )
    
    # Select optimal team
    optimal_team = selector.select_optimal_team(
        evaluations=evaluations,
        method=args.selection_method
    )
    
    # Save team selection
    save_team_selection(optimal_team, 'outputs/team_selection.txt')
    
    # Compare different selection methods
    print("\n" + "="*60)
    print("COMPARING SELECTION METHODS")
    print("="*60)
    comparison_df = compare_selection_methods(evaluations, model, device)
    print(comparison_df.to_string())
    comparison_df.to_csv('outputs/method_comparison.csv', index=False)
    
    # ==========================================
    # 8. GENERATE ADDITIONAL VISUALIZATIONS
    # ==========================================
    if args.save_plots:
        print_banner("STEP 8: GENERATING VISUALIZATIONS")
        
        # Player distribution plots
        plot_player_distribution(df, 'outputs/player_distribution.png')
        
        # Correlation matrix
        plot_correlation_matrix(
            df, 
            numerical_features[:10],  # Top 10 features for clarity
            'outputs/correlation_matrix.png'
        )
    
    # ==========================================
    # 9. LOG EXPERIMENT
    # ==========================================
    print_banner("EXPERIMENT COMPLETE")
    
    # Create experiment log
    experiment_metrics = {
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'test_position_accuracy': metrics['position_metrics']['accuracy'],
        'test_team_fit_r2': metrics['team_fit_metrics']['r2_score'],
        'optimal_team_score': optimal_team['team_metrics']['avg_overall_score']
    }
    
    create_experiment_log(
        experiment_name=f"NBA_ANN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        metrics=experiment_metrics
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Position Classification Accuracy: {metrics['position_metrics']['accuracy']:.4f}")
    print(f"Team Fit R² Score: {metrics['team_fit_metrics']['r2_score']:.4f}")
    print(f"Optimal Team Average Score: {optimal_team['team_metrics']['avg_overall_score']:.4f}")
    print(f"Team Composition: {optimal_team['composition_analysis']}")
    print("="*60)
    
    print("\n✅ Pipeline completed successfully!")
    print(f"📁 Results saved to 'outputs/' directory")
    print(f"🚀 To view interactive results, run: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()

    