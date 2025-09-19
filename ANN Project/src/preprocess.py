"""
Data Preprocessing Module
Handles normalization, scaling, and encoding of NBA player features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Tuple, Dict, Optional
import torch

class NBADataPreprocessor:
    """
    Preprocessor for NBA player data.
    Handles scaling, normalization, and encoding of features.
    """
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 handle_missing: str = 'mean'):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
            handle_missing: Strategy for handling missing values ('mean', 'median', 'drop')
        """
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.label_encoders = {}
        self.feature_means = {}
        self.feature_medians = {}
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, 
            numerical_features: list,
            categorical_features: list) -> None:
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            numerical_features: List of numerical feature columns
            categorical_features: List of categorical feature columns
        """
        # Handle missing values for numerical features
        for col in numerical_features:
            if col in df.columns:
                self.feature_means[col] = df[col].mean()
                self.feature_medians[col] = df[col].median()
        
        # Fit scaler on numerical features
        numerical_data = self._handle_missing_values(df[numerical_features].copy())
        self.scaler.fit(numerical_data)
        
        # Fit label encoders for categorical features
        for col in categorical_features:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values in categorical features
                df_col = df[col].fillna('unknown')
                self.label_encoders[col].fit(df_col)
        
        self.fitted = True
        print(f"Preprocessor fitted on {len(df)} samples")
        
    def transform(self, df: pd.DataFrame,
                  numerical_features: list,
                  categorical_features: list) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Transform the data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            numerical_features: List of numerical feature columns
            categorical_features: List of categorical feature columns
            
        Returns:
            Tuple of (processed_features, processed_dataframe)
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_processed = df.copy()
        
        # Process numerical features
        numerical_data = self._handle_missing_values(df_processed[numerical_features].copy())
        numerical_scaled = self.scaler.transform(numerical_data)
        
        # Process categorical features
        categorical_encoded = []
        for col in categorical_features:
            if col in df_processed.columns:
                # Handle unseen categories
                df_col = df_processed[col].fillna('unknown')
                df_col = df_col.map(lambda x: x if x in self.label_encoders[col].classes_ else 'unknown')
                
                # Ensure 'unknown' is in classes
                if 'unknown' not in self.label_encoders[col].classes_:
                    # Add 'unknown' to the encoder
                    encoded = np.zeros(len(df_col))
                else:
                    encoded = self.label_encoders[col].transform(df_col)
                categorical_encoded.append(encoded.reshape(-1, 1))
        
        # Combine features
        if categorical_encoded:
            categorical_array = np.hstack(categorical_encoded)
            processed_features = np.hstack([numerical_scaled, categorical_array])
        else:
            processed_features = numerical_scaled
        
        return processed_features, df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on specified strategy.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()  # Make a copy to avoid modifying the original
        
        if self.handle_missing == 'mean':
            for col in df.columns:
                if col in self.feature_means:
                    df[col] = df[col].fillna(self.feature_means[col])
        elif self.handle_missing == 'median':
            for col in df.columns:
                if col in self.feature_medians:
                    df[col] = df[col].fillna(self.feature_medians[col])
        elif self.handle_missing == 'drop':
            df = df.dropna()
        else:
            # Fill with 0 as default
            df = df.fillna(0)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame,
                      numerical_features: list,
                      categorical_features: list) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            numerical_features: List of numerical feature columns
            categorical_features: List of categorical feature columns
            
        Returns:
            Tuple of (processed_features, processed_dataframe)
        """
        self.fit(df, numerical_features, categorical_features)
        return self.transform(df, numerical_features, categorical_features)

def create_feature_tensors(features: np.ndarray, 
                          targets: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert numpy arrays to PyTorch tensors.
    
    Args:
        features: Feature array
        targets: Target array (optional)
        
    Returns:
        Tuple of (feature_tensor, target_tensor)
    """
    feature_tensor = torch.FloatTensor(features)
    
    if targets is not None:
        target_tensor = torch.FloatTensor(targets)
        return feature_tensor, target_tensor
    
    return feature_tensor, None

def normalize_targets(targets: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Normalize target values to 0-1 range.
    
    Args:
        targets: Target array
        
    Returns:
        Tuple of (normalized_targets, normalization_params)
    """
    min_val = targets.min(axis=0)
    max_val = targets.max(axis=0)
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    
    normalized = (targets - min_val) / range_val
    
    params = {
        'min': min_val,
        'max': max_val,
        'range': range_val
    }
    
    return normalized, params

def denormalize_targets(normalized_targets: np.ndarray, 
                       params: Dict) -> np.ndarray:
    """
    Denormalize targets back to original scale.
    
    Args:
        normalized_targets: Normalized target array
        params: Normalization parameters
        
    Returns:
        Denormalized targets
    """
    return normalized_targets * params['range'] + params['min']