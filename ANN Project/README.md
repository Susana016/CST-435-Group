# CST-435
# 🏀 NBA Team Selection Using Artificial Neural Networks

## Project Overview

This project implements a deep artificial neural network (ANN) system to select the optimal NBA team of 5 players from a pool of 100 players. The system uses a Multi-Layer Perceptron (MLP) architecture with forward propagation, backpropagation, and advanced team selection algorithms.

## 🎯 Objectives

1. **Analyze** NBA player statistics and characteristics
2. **Classify** players into optimal positions (Guard, Forward, Center)
3. **Evaluate** team fit scores for synergistic team composition
4. **Select** the optimal 5-player team using intelligent algorithms

## 🏗️ Architecture

### Neural Network Design

```
Input Layer (Player Features)
    ↓
Hidden Layer 1 (128 neurons) + ReLU + BatchNorm + Dropout
    ↓
Hidden Layer 2 (64 neurons) + ReLU + BatchNorm + Dropout
    ↓
Hidden Layer 3 (32 neurons) + ReLU + BatchNorm + Dropout
    ↓
Hidden Layer 4 (16 neurons) + ReLU + BatchNorm
    ↓
Output Layer (4 neurons)
    ├── Position Classification (3 neurons → Softmax)
    └── Team Fit Score (1 neuron → Sigmoid)
```

### Key Components

- **Forward Propagation**: Passes input through layers to generate predictions
- **Backpropagation**: Calculates gradients and updates weights
- **Custom Loss Function**: Combines CrossEntropy (position) and MSE (team fit)
- **Optimizer**: Adam with learning rate scheduling

## 📁 Project Structure

```
nba_ann_project/
│
├── data/
│   └── nba_players.csv          # NBA players dataset
│
├── src/
│   ├── load_data.py            # Data loading and filtering
│   ├── preprocess.py           # Feature preprocessing
│   ├── dataset.py              # PyTorch Dataset implementation
│   ├── model.py                # MLP architecture
│   ├── train.py                # Training loop implementation
│   ├── evaluate.py             # Model evaluation metrics
│   ├── select_team.py          # Team selection algorithms
│   └── utils.py                # Helper functions
│
├── app/
│   └── streamlit_app.py        # Interactive web interface
│
├── outputs/
│   ├── training_history.png    # Loss and accuracy curves
│   ├── confusion_matrix.png    # Position classification results
│   ├── team_selection.txt      # Optimal team results
│   └── evaluation_report.txt   # Comprehensive metrics
│
├── models/
│   ├── nba_model.pth           # Trained model weights
│   └── preprocessor.pkl        # Fitted preprocessor
│
├── docs/
│   └── report.md               # Technical report
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nba-ann-project.git
cd nba-ann-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

1. Place the NBA players dataset in `data/nba_players.csv`
2. Ensure the dataset contains the required columns:
   - Player statistics: pts, reb, ast, net_rating, etc.
   - Physical attributes: player_height, player_weight, age
   - Metadata: player_name, season, team_abbreviation

## 💻 Usage

### Option 1: Streamlit Web Interface (Recommended)

Run the interactive web application:

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: Python Script

```python
# Import modules
from src.load_data import load_nba_data, create_position_labels, split_data
from src.preprocess import NBADataPreprocessor
from src.dataset import create_data_loaders
from src.model import create_model
from src.train import Trainer
from src.evaluate import Evaluator
from src.select_team import TeamSelector

# 1. Load and prepare data
df = load_nba_data('data/nba_players.csv', start_year='2015-16', end_year='2019-20')
df = create_position_labels(df)
train_df, val_df, test_df = split_data(df)

# 2. Preprocess features
preprocessor = NBADataPreprocessor()
numerical_features = ['age', 'player_height', 'player_weight', 'gp', 'pts', 'reb', 'ast']
categorical_features = ['college', 'country']

train_features, _ = preprocessor.fit_transform(train_df, numerical_features, categorical_features)
val_features, _ = preprocessor.transform(val_df, numerical_features, categorical_features)
test_features, _ = preprocessor.transform(test_df, numerical_features, categorical_features)

# 3. Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    train_features, train_targets,
    val_features, val_targets,
    test_features, test_targets,
    batch_size=16
)

# 4. Build and train model
model = create_model(input_dim=train_features.shape[1])
trainer = Trainer(model, learning_rate=0.001)
history = trainer.train(train_loader, val_loader, epochs=50)

# 5. Evaluate model
evaluator = Evaluator(model)
metrics = evaluator.evaluate(test_loader)

# 6. Select optimal team
selector = TeamSelector(model)
evaluations = selector.evaluate_players(all_features, player_names, player_stats)
optimal_team = selector.select_optimal_team(evaluations, method='balanced')
```

## 📊 Model Performance

### Expected Results

- **Position Classification Accuracy**: ~85-90%
- **Team Fit R² Score**: ~0.75-0.85
- **Training Time**: ~5-10 minutes (CPU) / ~1-2 minutes (GPU)

### Evaluation Metrics

1. **Position Classification**:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix

2. **Team Fit Regression**:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² Score
   - Mean Absolute Error (MAE)

## 🎮 Team Selection Strategies

### 1. Greedy Selection
- Selects top 5 players by overall score
- Fast but may lack position balance

### 2. Balanced Selection
- Ensures position requirements:
  - 1-2 Guards
  - 2-3 Forwards
  - 1-2 Centers
- Provides well-rounded team composition

### 3. Exhaustive Search
- Evaluates multiple team combinations
- Optimizes for team synergy
- Most computationally intensive but produces best results

## 📈 Visualizations

The system generates various visualizations:

- Training loss and accuracy curves
- Position classification confusion matrix
- Team fit score predictions scatter plot
- Feature correlation heatmap
- Player statistics distributions

## 🌐 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Connect your GitHub repo to Streamlit Cloud
3. Deploy with one click

### Local Deployment

```bash
# Build Docker image (optional)
docker build -t nba-team-selector .

# Run container
docker run -p 8501:8501 nba-team-selector
```

## 📝 Technical Report

The comprehensive technical report includes:

1. **Problem Statement**: Detailed project objectives
2. **Algorithm Description**: Complete neural network implementation
3. **Analysis of Findings**: Performance metrics and insights
4. **References**: Academic and technical sources

Access the report in `docs/report.md` or through the Streamlit interface.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- NBA for providing player statistics
- PyTorch team for the deep learning framework
- Streamlit for the interactive web framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an academic project demonstrating the application of artificial neural networks to sports analytics. The team selections are based on statistical analysis and may not reflect actual NBA team dynamics.