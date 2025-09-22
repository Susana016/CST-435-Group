"""
Test Script to Verify Installation and Basic Functionality
Run this script to ensure all dependencies are installed correctly
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test if all required libraries can be imported."""
    print("Testing library imports...")
    
    libraries = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'torch': 'PyTorch',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'plotly': 'Plotly',
        'streamlit': 'Streamlit',
        'tqdm': 'TQDM'
    }
    
    success = True
    for lib, name in libraries.items():
        try:
            __import__(lib)
            print(f"  ✅ {name} imported successfully")
        except ImportError:
            print(f"  ❌ {name} not found - please install with: pip install {lib}")
            success = False
    
    return success

def test_pytorch():
    """Test PyTorch functionality."""
    print("\nTesting PyTorch functionality...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Check PyTorch version
        print(f"  PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"  ✅ CUDA is available - GPU: {torch.cuda.get_device_name(0)}")
            device = 'cuda'
        else:
            print("  ℹ️  CUDA not available - will use CPU")
            device = 'cpu'
        
        # Create a simple tensor operation
        x = torch.randn(10, 5).to(device)
        y = torch.randn(5, 3).to(device)
        z = torch.matmul(x, y)
        print(f"  ✅ Tensor operations working - Result shape: {z.shape}")
        
        # Create a simple neural network
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.fc(x)
        
        model = TestNet().to(device)
        output = model(torch.randn(2, 10).to(device))
        print(f"  ✅ Neural network creation working - Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PyTorch test failed: {str(e)}")
        return False

def test_project_structure():
    """Test if project directories exist."""
    print("\nTesting project structure...")
    
    import os
    
    directories = ['src', 'app', 'data', 'models', 'outputs', 'docs']
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"  ✅ Directory '{directory}' exists")
        else:
            print(f"  ⚠️  Directory '{directory}' not found - will be created when needed")
    
    return True

def test_module_imports():
    """Test if project modules can be imported."""
    print("\nTesting project module imports...")
    
    modules = [
        'src.load_data',
        'src.preprocess',
        'src.dataset',
        'src.model',
        'src.train',
        'src.evaluate',
        'src.select_team',
        'src.utils'
    ]
    
    success = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ Module '{module}' imported successfully")
        except ImportError as e:
            print(f"  ❌ Module '{module}' import failed: {str(e)}")
            success = False
    
    return success

def create_sample_data():
    """Create a small sample dataset for testing."""
    print("\nCreating sample data for testing...")
    
    try:
        import pandas as pd
        import numpy as np
        import os
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 20
        
        data = {
            'player_name': [f'Player_{i}' for i in range(n_samples)],
            'team_abbreviation': np.random.choice(['LAL', 'BOS', 'GSW', 'MIA'], n_samples),
            'age': np.random.randint(19, 35, n_samples),
            'player_height': np.random.normal(200, 10, n_samples),
            'player_weight': np.random.normal(100, 15, n_samples),
            'college': np.random.choice(['Duke', 'Kentucky', 'UCLA', 'UNC'], n_samples),
            'country': ['USA'] * n_samples,
            'draft_year': np.random.randint(2010, 2020, n_samples),
            'draft_round': np.random.randint(1, 3, n_samples),
            'draft_number': np.random.randint(1, 61, n_samples),
            'gp': np.random.randint(20, 82, n_samples),
            'pts': np.random.uniform(5, 30, n_samples),
            'reb': np.random.uniform(2, 15, n_samples),
            'ast': np.random.uniform(1, 12, n_samples),
            'net_rating': np.random.uniform(-10, 15, n_samples),
            'oreb_pct': np.random.uniform(0, 0.2, n_samples),
            'dreb_pct': np.random.uniform(0.05, 0.3, n_samples),
            'usg_pct': np.random.uniform(0.1, 0.35, n_samples),
            'ts_pct': np.random.uniform(0.4, 0.65, n_samples),
            'ast_pct': np.random.uniform(0.05, 0.4, n_samples),
            'season': np.random.choice(['2018-19', '2019-20'], n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv('data/sample_nba_players.csv', index=False)
        
        print(f"  ✅ Sample data created: data/sample_nba_players.csv")
        print(f"     Shape: {df.shape}")
        print(f"     Columns: {list(df.columns)[:5]}... (and {len(df.columns)-5} more)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Sample data creation failed: {str(e)}")
        return False

def run_quick_test():
    """Run a quick test of the main pipeline."""
    print("\nRunning quick pipeline test...")
    
    try:
        import torch
        import numpy as np
        from src.model import create_model
        
        # Create a simple model
        model = create_model(input_dim=10)
        print(f"  ✅ Model created with {model.count_parameters():,} parameters")
        
        # Test forward pass
        x = torch.randn(4, 10)
        output = model(x)
        print(f"  ✅ Forward pass successful - Output shape: {output.shape}")
        
        # Test loss calculation
        from src.train import CustomLoss
        criterion = CustomLoss()
        
        # Create dummy targets
        # Create 4 class indices between 0 and 2
        class_indices = torch.tensor([0, 1, 2, 0])  # example labels for 4 samples

        # Convert to one-hot with 3 classes
        one_hot = torch.nn.functional.one_hot(class_indices, num_classes=3).float()

        # Assign correctly
        targets = torch.zeros(4, 4)
        targets[:, :3] = one_hot
        targets[:, 3] = torch.rand(4)  # Team fit scores

        
        loss = criterion(output, targets)
        print(f"  ✅ Loss calculation successful - Loss value: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("NBA TEAM SELECTION ANN - INSTALLATION TEST")
    print("=" * 60)
    
    tests = [
        ("Library Imports", test_imports),
        ("PyTorch Functionality", test_pytorch),
        ("Project Structure", test_project_structure),
        ("Module Imports", test_module_imports),
        ("Sample Data Creation", create_sample_data),
        ("Quick Pipeline Test", run_quick_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        all_passed = all_passed and passed
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! The project is ready to run.")
        print("\nNext steps:")
        print("1. Place your NBA dataset in: data/nba_players.csv")
        print("2. Run the main pipeline: python main.py")
        print("3. Or launch the web app: streamlit run app/streamlit_app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the project.")
        print("\nTip: Install all dependencies with: pip install -r requirements.txt")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())