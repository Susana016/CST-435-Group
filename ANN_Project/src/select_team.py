"""
Optimal Team Selection Module
Uses the trained model to select the best 5-player team with balanced composition
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from itertools import combinations

class TeamSelector:
    """
    Selects optimal NBA team based on model predictions and team composition rules.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize team selector.
        
        Args:
            model: Trained model for player evaluation
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Team composition constraints
        self.ideal_composition = {
            'Guard': (1, 2),      # 1-2 guards
            'Forward': (2, 3),    # 2-3 forwards  
            'Center': (1, 2)      # 1-2 centers
        }
    
    def evaluate_players(self, 
                        features: np.ndarray,
                        player_names: List[str],
                        player_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all players using the model.
        
        Args:
            features: Player features array
            player_names: List of player names
            player_stats: Original player statistics
            
        Returns:
            DataFrame with player evaluations
        """
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
        
        # Get predictions
        position_probs = self.model.get_position_predictions(outputs).cpu().numpy()
        team_fit_scores = self.model.get_team_fit_score(outputs).cpu().numpy()
        
        # Create evaluation dataframe
        position_names = ['Guard', 'Forward', 'Center']
        predicted_positions = position_probs.argmax(axis=1)
        
        evaluations = pd.DataFrame({
            'player_name': player_names,
            'predicted_position': [position_names[pos] for pos in predicted_positions],
            'guard_prob': position_probs[:, 0],
            'forward_prob': position_probs[:, 1],
            'center_prob': position_probs[:, 2],
            'team_fit_score': team_fit_scores,
            'position_confidence': position_probs.max(axis=1)
        })
        
        # Add original stats for reference
        if player_stats is not None:
            evaluations = evaluations.merge(
                player_stats[['player_name', 'pts', 'reb', 'ast', 'net_rating']],
                on='player_name',
                how='left'
            )
        
        # Calculate overall player score
        evaluations['overall_score'] = (
            evaluations['team_fit_score'] * 0.5 +
            evaluations['position_confidence'] * 0.3 +
            evaluations[['pts', 'reb', 'ast']].sum(axis=1) / 50 * 0.2
        )
        
        return evaluations
    
    def select_optimal_team(self,
                           evaluations: pd.DataFrame,
                           method: str = 'greedy') -> Dict:
        """
        Select the optimal 5-player team.
        
        Args:
            evaluations: Player evaluations DataFrame
            method: Selection method ('greedy', 'balanced', 'exhaustive')
            
        Returns:
            Dictionary with selected team and metrics
        """
        if method == 'greedy':
            team = self._select_greedy(evaluations)
        elif method == 'balanced':
            team = self._select_balanced(evaluations)
        elif method == 'exhaustive':
            team = self._select_exhaustive(evaluations)
        else:
            team = self._select_greedy(evaluations)
        
        return team
    
    def _select_greedy(self, evaluations: pd.DataFrame) -> Dict:
        """
        Select team using greedy approach - top 5 by overall score.
        
        Args:
            evaluations: Player evaluations
            
        Returns:
            Selected team dictionary
        """
        # Sort by overall score and select top 5
        top_players = evaluations.nlargest(5, 'overall_score')
        
        return self._create_team_dict(top_players, evaluations, 'Greedy Selection')
    
    def _select_balanced(self, evaluations: pd.DataFrame) -> Dict:
        """
        Select team ensuring balanced position distribution.
        ALWAYS ensures at least one of each position.
        
        Args:
            evaluations: Player evaluations
            
        Returns:
            Selected team dictionary
        """
        selected_players = []
        remaining = evaluations.copy()
        
        # CRITICAL: Ensure at least one of each position
        required_positions = ['Guard', 'Forward', 'Center']
        
        # First, select the best player for each position (ensuring we have at least 1 of each)
        for position in required_positions:
            position_players = remaining[
                remaining['predicted_position'] == position
            ]
            
            if len(position_players) == 0:
                # If no players predicted for this position, find the player with highest probability
                if position == 'Guard':
                    best_player = remaining.nlargest(1, 'guard_prob')
                elif position == 'Forward':
                    best_player = remaining.nlargest(1, 'forward_prob')
                else:  # Center
                    best_player = remaining.nlargest(1, 'center_prob')
            else:
                # Select best player for this position
                best_player = position_players.nlargest(1, 'overall_score')
            
            selected_players.append(best_player)
            remaining = remaining[~remaining['player_name'].isin(best_player['player_name'])]
        
        # Now we have 3 players (one of each position), select 2 more
        # Prefer: 1 more guard/forward for balance
        remaining_spots = 2
        
        # Try to get one more guard or forward
        guards_forwards = remaining[
            remaining['predicted_position'].isin(['Guard', 'Forward'])
        ]
        
        if len(guards_forwards) > 0:
            next_player = guards_forwards.nlargest(1, 'overall_score')
            selected_players.append(next_player)
            remaining = remaining[~remaining['player_name'].isin(next_player['player_name'])]
            remaining_spots -= 1
        
        # Fill last spot with best remaining player
        if remaining_spots > 0 and len(remaining) > 0:
            best_remaining = remaining.nlargest(remaining_spots, 'overall_score')
            selected_players.append(best_remaining)
        
        team = pd.concat(selected_players)
        
        # Ensure we have exactly 5 players
        if len(team) > 5:
            team = team.nlargest(5, 'overall_score')
        
        return self._create_team_dict(team, evaluations, 'Balanced Selection (Position-Enforced)')
    
    def _select_exhaustive(self, evaluations: pd.DataFrame, max_combinations: int = 10000) -> Dict:
        """
        Select team by evaluating multiple combinations.
        
        Args:
            evaluations: Player evaluations
            max_combinations: Maximum combinations to evaluate
            
        Returns:
            Selected team dictionary
        """
        # Get top candidates to reduce search space
        top_candidates = evaluations.nlargest(15, 'overall_score')
        
        best_team = None
        best_score = -float('inf')
        
        # Generate and evaluate combinations
        player_indices = list(range(len(top_candidates)))
        n_combinations = 0
        
        for combo in combinations(player_indices, 5):
            if n_combinations >= max_combinations:
                break
                
            team = top_candidates.iloc[list(combo)]
            score = self._evaluate_team_composition(team)
            
            if score > best_score:
                best_score = score
                best_team = team
            
            n_combinations += 1
        
        print(f"Evaluated {n_combinations} team combinations")
        
        return self._create_team_dict(best_team, evaluations, 'Exhaustive Search')
    
    def _evaluate_team_composition(self, team: pd.DataFrame) -> float:
        """
        Evaluate team composition quality.
        
        Args:
            team: Team DataFrame
            
        Returns:
            Team composition score
        """
        # Check position distribution
        position_counts = team['predicted_position'].value_counts()
        
        # Position balance score
        balance_score = 1.0
        for position, (min_count, max_count) in self.ideal_composition.items():
            count = position_counts.get(position, 0)
            if count < min_count:
                balance_score *= 0.7  # Penalty for missing position
            elif count > max_count:
                balance_score *= 0.9  # Smaller penalty for excess
        
        # Overall team quality
        quality_score = team['overall_score'].mean()
        
        # Team fit synergy
        synergy_score = team['team_fit_score'].mean()
        
        # Combined score
        total_score = (
            quality_score * 0.5 +
            synergy_score * 0.3 +
            balance_score * 0.2
        )
        
        return total_score
    
    def _create_team_dict(self, 
                         team: pd.DataFrame,
                         all_evaluations: pd.DataFrame,
                         method_name: str) -> Dict:
        """
        Create team dictionary with detailed information.
        
        Args:
            team: Selected team DataFrame
            all_evaluations: All player evaluations
            method_name: Name of selection method
            
        Returns:
            Team dictionary
        """
        position_dist = team['predicted_position'].value_counts().to_dict()
        
        team_dict = {
            'method': method_name,
            'players': team[['player_name', 'predicted_position', 
                            'team_fit_score', 'overall_score']].to_dict('records'),
            'position_distribution': position_dist,
            'team_metrics': {
                'avg_team_fit_score': team['team_fit_score'].mean(),
                'avg_overall_score': team['overall_score'].mean(),
                'total_points': team['pts'].sum() if 'pts' in team.columns else 0,
                'total_rebounds': team['reb'].sum() if 'reb' in team.columns else 0,
                'total_assists': team['ast'].sum() if 'ast' in team.columns else 0,
                'avg_net_rating': team['net_rating'].mean() if 'net_rating' in team.columns else 0
            },
            'composition_analysis': self._analyze_composition(position_dist)
        }
        
        return team_dict
    
    def _analyze_composition(self, position_dist: Dict) -> str:
        """
        Analyze team composition and provide insights.
        
        Args:
            position_dist: Position distribution dictionary
            
        Returns:
            Composition analysis string
        """
        guards = position_dist.get('Guard', 0)
        forwards = position_dist.get('Forward', 0)
        centers = position_dist.get('Center', 0)
        
        if guards >= 2 and forwards >= 2 and centers >= 1:
            return "Well-balanced team with good position coverage"
        elif guards >= 3:
            return "Guard-heavy lineup - excellent ball handling and perimeter play"
        elif forwards >= 3:
            return "Forward-heavy lineup - versatile and strong on both ends"
        elif centers >= 2:
            return "Big lineup - dominant in the paint and rebounding"
        else:
            return "Unconventional lineup with unique strategic advantages"

def compare_selection_methods(evaluations: pd.DataFrame,
                             model: nn.Module,
                             device: str = 'cpu') -> pd.DataFrame:
    """
    Compare different team selection methods.
    
    Args:
        evaluations: Player evaluations
        model: Trained model
        device: Device for computation
        
    Returns:
        Comparison DataFrame
    """
    selector = TeamSelector(model, device)
    
    methods = ['greedy', 'balanced', 'exhaustive']
    comparisons = []
    
    for method in methods:
        team = selector.select_optimal_team(evaluations, method)
        
        comparison = {
            'Method': method,
            'Avg Team Fit': team['team_metrics']['avg_team_fit_score'],
            'Avg Overall Score': team['team_metrics']['avg_overall_score'],
            'Guards': team['position_distribution'].get('Guard', 0),
            'Forwards': team['position_distribution'].get('Forward', 0),
            'Centers': team['position_distribution'].get('Center', 0),
            'Analysis': team['composition_analysis']
        }
        comparisons.append(comparison)
    
    return pd.DataFrame(comparisons)

def save_team_selection(team: Dict, save_path: str = 'outputs/team_selection.txt'):
    """
    Save team selection to file.
    
    Args:
        team: Team selection dictionary
        save_path: Path to save the selection
    """
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("OPTIMAL NBA TEAM SELECTION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Selection Method: {team['method']}\n")
        f.write("-" * 40 + "\n\n")
        
        f.write("SELECTED PLAYERS:\n")
        for i, player in enumerate(team['players'], 1):
            f.write(f"{i}. {player['player_name']:<25} "
                   f"Position: {player['predicted_position']:<10} "
                   f"Team Fit: {player['team_fit_score']:.3f} "
                   f"Overall: {player['overall_score']:.3f}\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write("TEAM COMPOSITION:\n")
        for position, count in team['position_distribution'].items():
            f.write(f"  {position}s: {count}\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write("TEAM METRICS:\n")
        for metric, value in team['team_metrics'].items():
            f.write(f"  {metric.replace('_', ' ').title()}: {value:.3f}\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write(f"ANALYSIS: {team['composition_analysis']}\n")
        f.write("=" * 60 + "\n")
    
    print(f"Team selection saved to {save_path}")