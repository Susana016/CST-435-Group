"""
Launch script for Streamlit app
Run this from the project root directory
"""

import os
import sys
import subprocess

def main():
    """Launch the Streamlit app with correct paths."""
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Change to project root
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    # Check if data file exists
    data_path = 'data/nba_players.csv'
    if not os.path.exists(data_path):
        print(f"⚠️  Warning: Data file not found at {data_path}")
        print("Please ensure your NBA dataset is placed at: data/nba_players.csv")
        print()
    else:
        print(f"✅ Data file found at {data_path}")
    
    # Path to streamlit app
    app_path = os.path.join('app', 'streamlit_app.py')
    
    if not os.path.exists(app_path):
        print(f"❌ Error: Streamlit app not found at {app_path}")
        return 1
    
    print(f"✅ Streamlit app found at {app_path}")
    print()
    print("🚀 Launching Streamlit app...")
    print("=" * 60)
    
    # Launch streamlit
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path])
    except KeyboardInterrupt:
        print("\n\n👋 Streamlit app closed.")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())