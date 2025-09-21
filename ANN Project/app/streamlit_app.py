"""
NBA Team Selection - Streamlit Web Application
Interactive interface for model training, evaluation, and team selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Change working directory to project root for consistent file access
if os.getcwd() != project_root:
    os.chdir(project_root)

# Import project modules
from src.load_data import load_nba_data, get_feature_columns, create_position_labels, split_data
from src.preprocess import NBADataPreprocessor, create_feature_tensors, normalize_targets
from src.dataset import create_data_loaders
from src.model import create_model
from src.train import Trainer
from src.evaluate import Evaluator, plot_confusion_matrix, plot_team_fit_predictions
from src.select_team import TeamSelector, compare_selection_methods, save_team_selection
from src.utils import ensure_directories, save_model, load_model, get_device, set_random_seeds

# Page configuration
st.set_page_config(
    page_title="NBA Team Selection AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004E89;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .player-card {
        background-color: #e8f4ea;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None

def main():
    """Main application function."""
    
    # Title
    st.markdown('<h1 class="main-header">🏀 NBA Team Selection AI System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x150/FF6B35/FFFFFF?text=NBA+AI")
        st.markdown("---")
        
        page = st.selectbox(
            "Navigation",
            ["🏠 Home", "📊 Data Explorer", "🧠 Model Training", 
             "📈 Evaluation", "👥 Team Selection", "📝 Report"]
        )
        
        st.markdown("---")
        st.markdown("### Configuration")
        
        # Model parameters
        with st.expander("Model Parameters"):
            hidden_dims = st.text_input("Hidden Layers", "256,128,64,32")
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.25)
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.003, format="%.4f")
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=0)
        
        # Data parameters
        with st.expander("Data Parameters"):
            start_year = st.selectbox("Start Year", 
                                     ['1996-97','2015-16', '2016-17', '2017-18', '2018-19'],
                                     index=0)
            end_year = st.selectbox("End Year",
                                   ['2019-20', '2020-21', '2021-22', '2022-23'],
                                   index=0)
            n_players = st.slider("Number of Players", 50, 12000, 100)
    
    # Main content based on page selection
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Explorer":
        show_data_explorer(start_year, end_year, n_players)
    elif page == "🧠 Model Training":
        show_model_training(hidden_dims, dropout_rate, learning_rate, batch_size)
    elif page == "📈 Evaluation":
        show_evaluation()
    elif page == "👥 Team Selection":
        show_team_selection()
    elif page == "📝 Report":
        show_report()

def show_home_page():
    """Display home page."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the NBA Team Selection AI System
        
        This advanced artificial neural network system uses deep learning to:
        
        - **Analyze** NBA player statistics and characteristics
        - **Classify** players into optimal positions (Guard, Forward, Center)
        - **Evaluate** team fit scores for synergistic team composition
        - **Select** the optimal 5-player team using intelligent algorithms
        
        ### 🎯 Key Features
        
        - **Multi-Layer Perceptron (MLP)** with customizable architecture
        - **Forward & Backward Propagation** for learning
        - **Advanced Team Selection Algorithms**
        - **Interactive Visualizations** and analytics
        - **Real-time Model Training** and evaluation
        
        ### 🔬 Technical Implementation
        
        The system implements:
        - Deep neural network with multiple hidden layers
        - Backpropagation algorithm for weight optimization
        - Dropout and batch normalization for regularization
        - Custom loss function combining classification and regression
        - Multiple team selection strategies (Greedy, Balanced, Exhaustive)
        """)
    
    with col2:
        st.info("""
        **Quick Start Guide:**
        
        1. 📊 **Explore Data** - Analyze player statistics
        2. 🧠 **Train Model** - Build and train the ANN
        3. 📈 **Evaluate** - Assess model performance
        4. 👥 **Select Team** - Find optimal team
        5. 📝 **Generate Report** - Export results
        """)
        
        st.success("""
        **Model Architecture:**
        - Input Layer: Player features
        - Hidden Layers: 4 layers (128→64→32→16)
        - Output Layer: Position + Team Fit
        - Activation: ReLU
        - Optimizer: Adam
        """)

def show_data_explorer(start_year, end_year, n_players):
    """Display data explorer page."""
    
    st.markdown('<h2 class="sub-header">📊 NBA Players Data Explorer</h2>', 
                unsafe_allow_html=True)
    
    # Get project root for error messages
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load data button
    if st.button("Load NBA Data", type="primary"):
        with st.spinner("Loading data..."):
            try:
                # Check if data file exists
                data_path = 'data/nba_players.csv'
                if not os.path.exists(data_path):
                    st.error(f"""
                    Data file not found at: {os.path.abspath(data_path)}
                    
                    Please ensure the NBA dataset is in the correct location:
                    - The file should be at: {project_root}/data/nba_players.csv
                    
                    Current working directory: {os.getcwd()}
                    """)
                    return
                
                # Ensure directories exist
                ensure_directories()
                
                # Load data
                df = load_nba_data(data_path, 
                                 start_year, end_year, n_players)
                
                # Create position labels
                df = create_position_labels(df)
                
                # Store in session state
                st.session_state.data = df
                
                st.success(f"Loaded {len(df)} players from {start_year} to {end_year}")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Display data if loaded
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", len(df))
        with col2:
            st.metric("Avg Points", f"{df['pts'].mean():.1f}")
        with col3:
            st.metric("Avg Rebounds", f"{df['reb'].mean():.1f}")
        with col4:
            st.metric("Avg Assists", f"{df['ast'].mean():.1f}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Data Table", "Statistics", "Visualizations", "Correlations"])
        
        with tab1:
            st.subheader("Player Data")
            st.dataframe(df[['player_name', 'age', 'pts', 'reb', 'ast', 
                            'team_fit_score', 'primary_position']].head(20))
        
        with tab2:
            st.subheader("Statistical Summary")
            st.dataframe(df[['pts', 'reb', 'ast', 'net_rating', 
                            'ts_pct', 'usg_pct']].describe())
        
        with tab3:
            st.subheader("Data Visualizations")
            
            # Position distribution
            position_counts = df['primary_position'].map({0: 'Guard', 1: 'Forward', 2: 'Center'}).value_counts()
            fig_pos = px.pie(values=position_counts.values, 
                            names=position_counts.index,
                            title="Position Distribution")
            st.plotly_chart(fig_pos, key='position_dist')
            
            # Stats distribution
            fig_stats = go.Figure()
            fig_stats.add_trace(go.Box(y=df['pts'], name='Points'))
            fig_stats.add_trace(go.Box(y=df['reb'], name='Rebounds'))
            fig_stats.add_trace(go.Box(y=df['ast'], name='Assists'))
            fig_stats.update_layout(title="Player Statistics Distribution")
            st.plotly_chart(fig_stats, key='stats_dist')
        
        with tab4:
            st.subheader("Feature Correlations")
            
            # Correlation heatmap
            numerical_cols = ['pts', 'reb', 'ast', 'net_rating', 
                            'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']
            corr_matrix = df[numerical_cols].corr()
            
            fig_corr = px.imshow(corr_matrix, 
                               labels=dict(color="Correlation"),
                               x=numerical_cols, y=numerical_cols,
                               color_continuous_scale='RdBu',
                               zmin=-1, zmax=1)
            fig_corr.update_layout(title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

def show_model_training(hidden_dims, dropout_rate, learning_rate, batch_size):
    """Display model training page."""
    
    st.markdown('<h2 class="sub-header">🧠 Neural Network Training</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please load data first in the Data Explorer page!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        epochs = st.slider("Training Epochs", 10, 500, 200)
        
        if st.button("Start Training", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Prepare data
                    df = st.session_state.data
                    numerical_features, categorical_features, _ = get_feature_columns()
                    
                    # Split data with stratification
                    train_df, val_df, test_df = split_data(df)
                    
                    # Preprocess
                    preprocessor = NBADataPreprocessor()
                    train_features, _ = preprocessor.fit_transform(train_df, numerical_features, categorical_features)
                    val_features, _ = preprocessor.transform(val_df, numerical_features, categorical_features)
                    test_features, _ = preprocessor.transform(test_df, numerical_features, categorical_features)
                    
                    # Prepare targets
                    train_targets = train_df[['guard_score', 'forward_score', 'center_score', 'team_fit_score']].values
                    val_targets = val_df[['guard_score', 'forward_score', 'center_score', 'team_fit_score']].values
                    test_targets = test_df[['guard_score', 'forward_score', 'center_score', 'team_fit_score']].values
                    
                    # Normalize position scores to create one-hot encoding
                    for targets in [train_targets, val_targets, test_targets]:
                        targets[:, :3] = (targets[:, :3] == targets[:, :3].max(axis=1, keepdims=True)).astype(float)
                    
                    # Compute class weights for balanced training
                    from src.utils import compute_class_weights
                    class_weights = compute_class_weights(train_targets)
                    
                    # Get device
                    device = get_device()
                    
                    # Create data loaders
                    train_loader, val_loader, test_loader = create_data_loaders(
                        train_features, train_targets,
                        val_features, val_targets,
                        test_features, test_targets,
                        train_names=train_df['player_name'].tolist(),
                        val_names=val_df['player_name'].tolist(),
                        test_names=test_df['player_name'].tolist(),
                        batch_size=batch_size
                    )
                    
                    # Create model with stronger regularization
                    hidden_layers = [int(x) for x in hidden_dims.split(',')]
                    model = create_model(
                        input_dim=train_features.shape[1],
                        config={
                            'hidden_dims': hidden_layers,
                            'dropout_rate': dropout_rate,  # Use specified dropout
                            'activation': 'relu',
                            'use_batch_norm': True
                        }
                    )
                    
                    # Create trainer with class weights
                    trainer = Trainer(
                        model, device, 
                        learning_rate=learning_rate,
                        weight_decay=0.00005,  # Less aggressive regularization
                        patience=25,  # More patience before stopping
                        scheduler_type='plateau',  # Better scheduler
                        class_weights=class_weights
                    )
                    
                    # Training progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Custom training loop with progress updates
                    history = trainer.train(train_loader, val_loader, epochs=epochs)
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.trainer = trainer
                    st.session_state.preprocessor = preprocessor
                    st.session_state.training_history = history
                    st.session_state.test_loader = test_loader
                    
                    st.success("Training completed successfully!")
                    
                    # Display training results
                    st.subheader("Training Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
                        st.metric("Final Train Accuracy", f"{history['train_acc'][-1]:.4f}")
                    with col2:
                        st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
                        st.metric("Final Val Accuracy", f"{history['val_acc'][-1]:.4f}")
                    
                    # Plot training curves
                    fig_history = go.Figure()
                    fig_history.add_trace(go.Scatter(y=history['train_loss'], name='Train Loss'))
                    fig_history.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss'))
                    fig_history.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig_history, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
    
    with col2:
        if st.session_state.training_history:
            st.info("""
            **Training Status:** ✅ Complete
            
            **Model Architecture:**
            - Input: Player features
            - Hidden: Custom layers
            - Output: Position + Team Fit
            
            **Training Config:**
            - Optimizer: Adam
            - Loss: Custom (CE + MSE)
            - Regularization: Dropout + BN
            """)

def show_evaluation():
    """Display model evaluation page."""
    
    st.markdown('<h2 class="sub-header">📈 Model Evaluation</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train the model first!")
        return
    
    if st.button("Run Evaluation", type="primary"):
        with st.spinner("Evaluating model..."):
            try:
                # Create evaluator
                device = get_device()
                evaluator = Evaluator(st.session_state.model, device)
                
                # Evaluate on test set
                metrics = evaluator.evaluate(st.session_state.test_loader)
                
                # Display metrics
                st.subheader("Performance Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Position Classification")
                    st.metric("Accuracy", f"{metrics['position_metrics']['accuracy']:.4f}")
                    
                    # Classification report
                    report = metrics['position_metrics']['classification_report']
                    for position in ['Guard', 'Forward', 'Center']:
                        with st.expander(f"{position} Metrics"):
                            st.write(f"Precision: {report[position]['precision']:.3f}")
                            st.write(f"Recall: {report[position]['recall']:.3f}")
                            st.write(f"F1-Score: {report[position]['f1-score']:.3f}")
                
                with col2:
                    st.markdown("### Team Fit Regression")
                    st.metric("R² Score", f"{metrics['team_fit_metrics']['r2_score']:.4f}")
                    st.metric("RMSE", f"{metrics['team_fit_metrics']['rmse']:.4f}")
                    st.metric("MAE", f"{metrics['team_fit_metrics']['mae']:.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = metrics['position_metrics']['confusion_matrix']
                fig_cm = px.imshow(cm, 
                                 labels=dict(x="Predicted", y="True", color="Count"),
                                 x=['Guard', 'Forward', 'Center'],
                                 y=['Guard', 'Forward', 'Center'],
                                 text_auto=True)
                fig_cm.update_layout(title="Position Classification Confusion Matrix")
                st.plotly_chart(fig_cm, key='conf_matrix')
                
                # Team Fit Scatter
                st.subheader("Team Fit Score Predictions")
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=metrics['team_fit_metrics']['true_scores'],
                    y=metrics['team_fit_metrics']['predictions'],
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=8, opacity=0.6)
                ))
                fig_scatter.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                fig_scatter.update_layout(
                    title=f"Team Fit Predictions (R²={metrics['team_fit_metrics']['r2_score']:.3f})",
                    xaxis_title="True Score",
                    yaxis_title="Predicted Score"
                )
                st.plotly_chart(fig_scatter, key='fit_scatter')
                
            except Exception as e:
                st.error(f"Evaluation error: {str(e)}")

def show_team_selection():
    """Display team selection page."""
    
    st.markdown('<h2 class="sub-header">👥 Optimal Team Selection</h2>', 
                unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train the model first!")
        return
    
    selection_method = st.selectbox(
        "Selection Method",
        ["Greedy (Top 5)", "Balanced (Position-aware)", "Exhaustive Search"]
    )
    
    method_map = {
        "Greedy (Top 5)": "greedy",
        "Balanced (Position-aware)": "balanced",
        "Exhaustive Search": "exhaustive"
    }
    
    if st.button("Select Optimal Team", type="primary"):
        with st.spinner("Selecting optimal team..."):
            try:
                # Prepare data
                df = st.session_state.data
                numerical_features, categorical_features, _ = get_feature_columns()
                
                # Preprocess all data
                features, _ = st.session_state.preprocessor.transform(df, numerical_features, categorical_features)
                
                # Create team selector
                device = get_device()
                selector = TeamSelector(st.session_state.model, device)
                
                # Evaluate all players
                evaluations = selector.evaluate_players(features, df['player_name'].tolist(), df)
                
                # Select optimal team
                team = selector.select_optimal_team(evaluations, method=method_map[selection_method])
                
                # Display team
                st.success(f"Optimal team selected using {selection_method}")
                
                st.subheader("🏆 Selected Team")
                
                # Team composition
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Guards", team['position_distribution'].get('Guard', 0))
                with col2:
                    st.metric("Forwards", team['position_distribution'].get('Forward', 0))
                with col3:
                    st.metric("Centers", team['position_distribution'].get('Center', 0))
                
                # Team analysis
                st.info(f"**Team Analysis:** {team['composition_analysis']}")
                
                # Player cards
                st.subheader("Team Roster")
                for i, player in enumerate(team['players'], 1):
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        with col1:
                            st.markdown(f"**{i}. {player['player_name']}**")
                        with col2:
                            st.write(f"Position: {player['predicted_position']}")
                        with col3:
                            st.write(f"Team Fit: {player['team_fit_score']:.3f}")
                        with col4:
                            st.write(f"Overall: {player['overall_score']:.3f}")
                
                # Team metrics
                st.subheader("Team Metrics")
                metrics_df = pd.DataFrame([team['team_metrics']]).T
                metrics_df.columns = ['Value']
                st.dataframe(metrics_df)
                
                # Comparison of methods
                st.subheader("Method Comparison")
                comparison_df = compare_selection_methods(evaluations, st.session_state.model, device)
                st.dataframe(comparison_df)
                
            except Exception as e:
                st.error(f"Team selection error: {str(e)}")

def show_report():
    """Display report generation page."""
    
    st.markdown('<h2 class="sub-header">📝 Technical Report</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ## NBA Team Selection Using Artificial Neural Networks
    
    ### Problem Statement
    
    The objective of this project is to develop an artificial neural network (ANN) system capable of:
    1. Analyzing NBA player statistics and characteristics
    2. Classifying players into optimal positions
    3. Evaluating team synergy through team fit scores
    4. Selecting the optimal 5-player team from a pool of 100 players
    
    ### Algorithm of the Solution
    
    #### 1. Data Preprocessing
    - Loaded NBA player data for a 5-year window
    - Normalized numerical features using StandardScaler
    - Encoded categorical variables
    - Created synthetic position labels based on player statistics
    - Split data into train (70%), validation (15%), and test (15%) sets
    
    #### 2. Neural Network Architecture
    
    **Multi-Layer Perceptron (MLP) Design:**
    - **Input Layer:** Receives player features (statistics, physical attributes)
    - **Hidden Layers:** 4 fully connected layers (128 → 64 → 32 → 16 neurons)
    - **Activation:** ReLU for non-linearity
    - **Regularization:** Dropout (0.2) and Batch Normalization
    - **Output Layer:** 4 neurons (3 for position classification + 1 for team fit score)
    
    #### 3. Training Process
    
    **Forward Propagation:**
    1. Input features pass through hidden layers
    2. Activation functions introduce non-linearity
    3. Output layer produces position logits and team fit score
    
    **Loss Calculation:**
    - Combined loss = 0.6 × CrossEntropy(position) + 0.4 × MSE(team_fit)
    
    **Backpropagation:**
    1. Compute gradients of loss with respect to weights
    2. Use chain rule to propagate errors backward
    3. Update weights using Adam optimizer
    4. Apply gradient clipping to prevent exploding gradients
    
    #### 4. Team Selection Algorithm
    
    Three selection strategies implemented:
    1. **Greedy:** Select top 5 players by overall score
    2. **Balanced:** Ensure position requirements (1-2 guards, 2-3 forwards, 1-2 centers)
    3. **Exhaustive:** Evaluate multiple combinations for optimal synergy
    
    ### Analysis of Findings
    
    The trained ANN successfully:
    - Achieved high accuracy in position classification
    - Produced reliable team fit scores (high R² score)
    - Selected balanced teams with good position coverage
    - Demonstrated learning through decreasing loss curves
    
    Key insights:
    - Position classification benefits from player statistics patterns
    - Team fit scores correlate with overall player performance
    - Balanced selection method produces most versatile teams
    - The model generalizes well to unseen test data
    
    ### Conclusions
    
    This project successfully demonstrates the application of artificial neural networks to complex decision-making in sports analytics. The MLP architecture effectively learns player patterns and team dynamics, enabling intelligent team selection that considers both individual performance and team synergy.
    
    ### References
    
    1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
    2. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
    3. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.
    4. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
    5. PyTorch Documentation. (2024). Neural Network Tutorial. Retrieved from https://pytorch.org/tutorials/
    """)
    
    if st.button("Download Report", type="primary"):
        report_path = "docs/report.md"
        if os.path.exists(report_path):
            report_content = open(report_path).read()
        else:
            report_content = "Report not generated yet. Please run the main pipeline first."
        
        st.download_button(
            label="Download as Markdown",
            data=report_content,
            file_name="nba_team_selection_report.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()