"""
NBA Players Dataset Loading Module
This module handles loading and initial filtering of the NBA dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

def load_nba_data(filepath: str, 
                  start_year: Optional[str] = '2015-16',
                  end_year: Optional[str] = '2019-20',
                  n_players: int = 100,
                  random_state: int = 42) -> pd.DataFrame:
    """
    Load NBA players dataset and filter based on specified criteria.
    
    Args:
        filepath: Path to the NBA players CSV file
        start_year: Starting season for the 5-year window
        end_year: Ending season for the 5-year window
        n_players: Number of players to select from the pool
        random_state: Random seed for reproducibility
    
    Returns:
        Filtered DataFrame containing selected NBA players
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Filter by season (5-year window)
    year_mask = (df['season'] >= start_year) & (df['season'] <= end_year)
    df_filtered = df[year_mask].copy()
    
    # Remove duplicate players (keep the most recent season data)
    df_filtered = df_filtered.sort_values('season', ascending=False)
    df_filtered = df_filtered.drop_duplicates(subset=['player_name'], keep='first')
    
    # Filter players with sufficient game participation (at least 20 games)
    df_filtered = df_filtered[df_filtered['gp'] >= 20]
    
    # Select top players based on a composite score
    # Composite score considers: points, rebounds, assists, and efficiency
    df_filtered['composite_score'] = (
        df_filtered['pts'] * 1.0 +
        df_filtered['reb'] * 1.2 +
        df_filtered['ast'] * 1.5 +
        df_filtered['net_rating'] * 0.1
    )
    
    # Sort by composite score and select top N players
    df_filtered = df_filtered.nlargest(min(n_players, len(df_filtered)), 'composite_score')
    
    # Reset index
    df_filtered.reset_index(drop=True, inplace=True)
    
    print(f"Loaded {len(df_filtered)} players from seasons {start_year} to {end_year}")
    print(f"Columns available: {df_filtered.columns.tolist()}")
    
    return df_filtered

def get_feature_columns() -> Tuple[list, list, list]:
    """
    Define feature columns for the model.
    
    Returns:
        Tuple of (numerical_features, categorical_features, target_features)
    """
    numerical_features = [
        'age', 'player_height', 'player_weight', 'gp', 'pts', 
        'reb', 'ast', 'net_rating', 'oreb_pct', 'dreb_pct', 
        'usg_pct', 'ts_pct', 'ast_pct'
    ]
    
    categorical_features = ['college', 'country', 'draft_round']
    
    # Define position categories for team composition
    # We'll create synthetic labels based on player statistics
    target_features = ['position_score', 'team_fit_score']
    
    return numerical_features, categorical_features, target_features

def create_position_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create position labels based on player statistics.
    This helps in building a balanced team.
    
    Args:
        df: DataFrame with player statistics
    
    Returns:
        DataFrame with added position labels
    """
    df = df.copy()
    
    # Create position scores based on player characteristics
    # Guards: High assists, good shooting percentage, lower height/weight
    df['guard_score'] = (
        df['ast'] * 3.0 +  # Assists are most important for guards
        df['ts_pct'] * 50 +  # Shooting percentage
        df['ast_pct'] * 50 +  # Assist percentage
        (200 - df['player_height']) * 0.1 +  # Smaller players tend to be guards
        df['pts'] * 0.5  # Scoring ability
    )
    
    # Forwards: Balanced scoring and rebounding, medium height
    df['forward_score'] = (
        df['pts'] * 1.5 +  # Scoring
        df['reb'] * 2.0 +  # Rebounding
        df['usg_pct'] * 30 +  # Usage rate
        np.abs(df['player_height'] - 200) * -0.2 +  # Medium height preferred
        df['net_rating'] * 0.5  # Overall effectiveness
    )
    
    # Centers: High rebounding, good defensive rating, taller/heavier
    df['center_score'] = (
        df['reb'] * 3.0 +  # Rebounding is key for centers
        df['dreb_pct'] * 100 +  # Defensive rebounding percentage
        df['oreb_pct'] * 100 +  # Offensive rebounding percentage
        df['player_height'] * 0.3 +  # Height advantage
        df['player_weight'] * 0.2 +  # Weight/strength
        (10 - df['ast']) * 0.5  # Centers typically have fewer assists
    )
    
    # Normalize scores to 0-1 range for each position
    for col in ['guard_score', 'forward_score', 'center_score']:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
    
    # Determine primary position based on highest score
    position_scores = df[['guard_score', 'forward_score', 'center_score']]
    df['primary_position'] = position_scores.idxmax(axis=1).map({
        'guard_score': 0,  # Guard
        'forward_score': 1,  # Forward
        'center_score': 2  # Center
    })
    
    # Ensure balanced distribution by reassigning some players if needed
    position_counts = df['primary_position'].value_counts()
    target_count = len(df) // 3  # Aim for roughly equal distribution
    
    # If any position is underrepresented, reassign some borderline players
    for pos in [0, 1, 2]:
        if pos not in position_counts or position_counts[pos] < target_count * 0.5:
            # Find players who could play this position (high score but not primary)
            if pos == 0:  # Guard
                candidates = df[df['primary_position'] != 0].nlargest(target_count - position_counts.get(pos, 0), 'guard_score')
            elif pos == 1:  # Forward
                candidates = df[df['primary_position'] != 1].nlargest(target_count - position_counts.get(pos, 0), 'forward_score')
            else:  # Center
                candidates = df[df['primary_position'] != 2].nlargest(target_count - position_counts.get(pos, 0), 'center_score')
            
            # Reassign top candidates to this position
            df.loc[candidates.index[:max(5, target_count//4)], 'primary_position'] = pos
    
    # Team fit score (how well a player fits in a balanced team)
    df['team_fit_score'] = (
        df['pts'] * 0.25 +
        df['reb'] * 0.25 +
        df['ast'] * 0.25 +
        df['net_rating'] * 0.05 +
        df['ts_pct'] * 20  # Shooting efficiency
    )
    
    # Normalize team fit score to 0-1 range
    df['team_fit_score'] = (df['team_fit_score'] - df['team_fit_score'].min()) / \
                           (df['team_fit_score'].max() - df['team_fit_score'].min())
    
    # Print position distribution
    print(f"Position distribution: Guards={sum(df['primary_position']==0)}, "
          f"Forwards={sum(df['primary_position']==1)}, Centers={sum(df['primary_position']==2)}")
    
    return df

def split_data(df: pd.DataFrame, 
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets with stratification.
    
    Args:
        df: DataFrame to split
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    np.random.seed(random_state)
    
    # Stratified split to maintain position distribution
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_ratio = train_ratio + val_ratio
    
    if 'primary_position' in df.columns:
        # Use stratification if position labels exist
        train_val_df, test_df = train_test_split(
            df, test_size=1-train_val_ratio, 
            stratify=df['primary_position'],
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size = val_ratio / train_val_ratio
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size,
            stratify=train_val_df['primary_position'],
            random_state=random_state
        )
    else:
        # Random split if no position labels
        n = len(df)
        indices = np.random.permutation(n)
        
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)
    
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Print position distribution in each set
    if 'primary_position' in df.columns:
        for name, subset in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            counts = subset['primary_position'].value_counts().sort_index()
            print(f"{name} positions: Guards={counts.get(0, 0)}, "
                  f"Forwards={counts.get(1, 0)}, Centers={counts.get(2, 0)}")
    
    return train_df, val_df, test_df