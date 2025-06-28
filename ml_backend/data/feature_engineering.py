"""
Advanced Feature Engineering for IPL Predictions
Creates sophisticated features using domain knowledge and statistical methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for IPL match prediction"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.team_stats = {}
        self.venue_stats = {}
        
    async def create_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Create comprehensive feature set from match data"""
        features = {}
        
        # Basic match features
        features.update(self._create_basic_features(match_data))
        
        # Advanced cricket-specific features
        features.update(self._create_cricket_features(match_data))
        
        # Pressure and momentum features
        features.update(self._create_pressure_features(match_data))
        
        # Team and venue features
        features.update(self._create_team_venue_features(match_data))
        
        # Weather and conditions features
        features.update(self._create_conditions_features(match_data))
        
        # Time-based features
        features.update(self._create_temporal_features(match_data))
        
        # Statistical features
        features.update(self._create_statistical_features(match_data))
        
        logger.info(f"Created {len(features)} features")
        return features
    
    def _create_basic_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Create basic match state features"""
        features = {}
        
        # Direct features
        features['current_score'] = float(data.get('current_score', 0))
        features['wickets'] = float(data.get('wickets', 0))
        features['overs'] = float(data.get('overs', 0))
        features['balls'] = float(data.get('balls', 0))
        features['target'] = float(data.get('target', 0))
        
        # Derived features
        total_balls = data.get('overs', 0) * 6 + data.get('balls', 0)
        features['total_balls_faced'] = float(total_balls)
        features['balls_remaining'] = float(120 - total_balls)
        features['wickets_in_hand'] = float(10 - data.get('wickets', 0))
        
        # Rates
        if total_balls > 0:
            features['current_run_rate'] = (data.get('current_score', 0) / total_balls) * 6
        else:
            features['current_run_rate'] = 0.0
        
        if data.get('target', 0) > 0 and features['balls_remaining'] > 0:
            runs_required = data.get('target', 0) - data.get('current_score', 0)
            features['required_run_rate'] = (runs_required / features['balls_remaining']) * 6
            features['runs_required'] = float(runs_required)
        else:
            features['required_run_rate'] = 0.0
            features['runs_required'] = 0.0
        
        return features
    
    def _create_cricket_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Create cricket-specific features"""
        features = {}
        
        # Match phase
        balls_remaining = 120 - (data.get('overs', 0) * 6 + data.get('balls', 0))
        
        if balls_remaining > 96:  # First 4 overs
            features['phase_powerplay'] = 1.0
            features['phase_middle'] = 0.0
            features['phase_death'] = 0.0
        elif balls_remaining > 36:  # Overs 5-14
            features['phase_powerplay'] = 0.0
            features['phase_middle'] = 1.0
            features['phase_death'] = 0.0
        else:  # Last 6 overs
            features['phase_powerplay'] = 0.0
            features['phase_middle'] = 0.0
            features['phase_death'] = 1.0
        
        # Batting position strength
        wickets = data.get('wickets', 0)
        if wickets <= 2:
            features['batting_position_strength'] = 1.0  # Top order
        elif wickets <= 5:
            features['batting_position_strength'] = 0.7  # Middle order
        elif wickets <= 7:
            features['batting_position_strength'] = 0.4  # Lower middle
        else:
            features['batting_position_strength'] = 0.2  # Tail
        
        # Over progression features
        overs_completed = data.get('overs', 0)
        features['overs_completed_normalized'] = overs_completed / 20.0
        features['overs_remaining_normalized'] = (20 - overs_completed) / 20.0
        
        return features
    
    def _create_pressure_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Create pressure and momentum features"""
        features = {}
        
        current_rr = features.get('current_run_rate', 0)
        required_rr = features.get('required_run_rate', 0)
        
        # Run rate pressure
        if current_rr > 0:
            features['run_rate_pressure'] = required_rr / (current_rr + 0.1)
        else:
            features['run_rate_pressure'] = required_rr / 6.0
        
        # Wicket pressure
        wickets_lost = data.get('wickets', 0)
        features['wicket_pressure'] = wickets_lost / 10.0
        
        # Balls pressure (time running out)
        balls_remaining = 120 - (data.get('overs', 0) * 6 + data.get('balls', 0))
        features['time_pressure'] = 1.0 - (balls_remaining / 120.0)
        
        # Combined pressure index
        features['pressure_index'] = (
            features['run_rate_pressure'] * 0.4 +
            features['wicket_pressure'] * 0.3 +
            features['time_pressure'] * 0.3
        )
        
        # Momentum score
        wickets_in_hand = 10 - wickets_lost
        features['momentum_score'] = (
            current_rr * 0.3 +
            wickets_in_hand * 0.4 +
            (balls_remaining / 120.0) * 0.3
        )
        
        return features
    
    def _create_team_venue_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Create team and venue specific features"""
        features = {}
        
        # Team encoding
        team_mapping = {
            'Chennai Super Kings': 0, 'Mumbai Indians': 1, 'Royal Challengers Bangalore': 2,
            'Kolkata Knight Riders': 3, 'Delhi Capitals': 4, 'Rajasthan Royals': 5,
            'Punjab Kings': 6, 'Sunrisers Hyderabad': 7
        }
        
        features['team1_encoded'] = float(team_mapping.get(data.get('team1', ''), 0))
        features['team2_encoded'] = float(team_mapping.get(data.get('team2', ''), 0))
        
        # Venue encoding
        venue_mapping = {
            'Wankhede Stadium, Mumbai': 0, 'Eden Gardens, Kolkata': 1,
            'M. Chinnaswamy Stadium, Bangalore': 2, 'MA Chidambaram Stadium, Chennai': 3,
            'Arun Jaitley Stadium, Delhi': 4, 'Sawai Mansingh Stadium, Jaipur': 5,
            'PCA Stadium, Mohali': 6, 'Rajiv Gandhi International Stadium, Hyderabad': 7
        }
        
        features['venue_encoded'] = float(venue_mapping.get(data.get('venue', ''), 0))
        
        # Home advantage
        venue_team_map = {
            'Wankhede Stadium, Mumbai': 'Mumbai Indians',
            'Eden Gardens, Kolkata': 'Kolkata Knight Riders',
            'M. Chinnaswamy Stadium, Bangalore': 'Royal Challengers Bangalore',
            'MA Chidambaram Stadium, Chennai': 'Chennai Super Kings',
            'Arun Jaitley Stadium, Delhi': 'Delhi Capitals',
            'Sawai Mansingh Stadium, Jaipur': 'Rajasthan Royals',
            'PCA Stadium, Mohali': 'Punjab Kings',
            'Rajiv Gandhi International Stadium, Hyderabad': 'Sunrisers Hyderabad'
        }
        
        home_team = venue_team_map.get(data.get('venue', ''), '')
        features['team1_home_advantage'] = 1.0 if data.get('team1') == home_team else 0.0
        features['team2_home_advantage'] = 1.0 if data.get('team2') == home_team else 0.0
        
        # Toss advantage
        toss_winner = data.get('toss_winner', '')
        toss_decision = data.get('toss_decision', '')
        
        features['toss_winner_team1'] = 1.0 if toss_winner == data.get('team1') else 0.0
        features['toss_decision_bat'] = 1.0 if toss_decision == 'bat' else 0.0
        features['toss_advantage'] = 1.0 if (
            (toss_winner == data.get('team1') and toss_decision == 'bat') or
            (toss_winner == data.get('team2') and toss_decision == 'bowl')
        ) else 0.0
        
        return features
    
    def _create_conditions_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Create weather and playing conditions features"""
        features = {}
        
        # Weather encoding
        weather_mapping = {
            'Clear': 0, 'Partly Cloudy': 1, 'Overcast': 2, 'Light Rain': 3, 'Dew Expected': 4
        }
        
        features['weather_encoded'] = float(weather_mapping.get(data.get('weather', ''), 0))
        
        # Weather impact on batting/bowling
        weather = data.get('weather', '')
        if weather in ['Clear', 'Partly Cloudy']:
            features['batting_friendly_weather'] = 1.0
            features['bowling_friendly_weather'] = 0.0
        elif weather in ['Overcast', 'Light Rain']:
            features['batting_friendly_weather'] = 0.0
            features['bowling_friendly_weather'] = 1.0
        else:
            features['batting_friendly_weather'] = 0.5
            features['bowling_friendly_weather'] = 0.5
        
        # Dew factor (helps chasing team)
        features['dew_factor'] = 1.0 if weather == 'Dew Expected' else 0.0
        
        # Conditions score
        features['conditions_score'] = (
            features['batting_friendly_weather'] * 0.6 +
            (1.0 - features['dew_factor']) * 0.4
        ) * 10.0
        
        return features
    
    def _create_temporal_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Create time-based features"""
        features = {}
        
        # Match progression
        total_balls = data.get('overs', 0) * 6 + data.get('balls', 0)
        features['match_progress'] = total_balls / 120.0
        
        # Critical phases
        features['is_powerplay'] = 1.0 if total_balls <= 36 else 0.0
        features['is_middle_overs'] = 1.0 if 36 < total_balls <= 96 else 0.0
        features['is_death_overs'] = 1.0 if total_balls > 96 else 0.0
        
        # Time criticality
        if data.get('target', 0) > 0:
            balls_remaining = 120 - total_balls
            features['time_criticality'] = 1.0 - (balls_remaining / 120.0)
        else:
            features['time_criticality'] = features['match_progress']
        
        return features
    
    def _create_statistical_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Create statistical and derived features"""
        features = {}
        
        # Score efficiency
        current_score = data.get('current_score', 0)
        total_balls = data.get('overs', 0) * 6 + data.get('balls', 0)
        
        if total_balls > 0:
            features['score_efficiency'] = current_score / total_balls
        else:
            features['score_efficiency'] = 0.0
        
        # Wicket efficiency
        wickets = data.get('wickets', 0)
        if total_balls > 0:
            features['wicket_rate'] = wickets / total_balls
        else:
            features['wicket_rate'] = 0.0
        
        # Target achievement probability (simple heuristic)
        if data.get('target', 0) > 0:
            runs_required = data.get('target', 0) - current_score
            balls_remaining = 120 - total_balls
            
            if balls_remaining > 0:
                required_rate = (runs_required / balls_remaining) * 6
                features['target_difficulty'] = required_rate / 12.0  # Normalize by max reasonable RR
            else:
                features['target_difficulty'] = 1.0 if runs_required > 0 else 0.0
        else:
            features['target_difficulty'] = 0.0
        
        # Form indicators (simplified)
        features['team_form_score'] = np.random.uniform(6, 9)  # Would be calculated from historical data
        features['recent_performance'] = np.random.uniform(0.4, 0.8)  # Would be calculated from recent matches
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # This would return all possible feature names
        return [
            'current_score', 'wickets', 'overs', 'balls', 'target',
            'total_balls_faced', 'balls_remaining', 'wickets_in_hand',
            'current_run_rate', 'required_run_rate', 'runs_required',
            'phase_powerplay', 'phase_middle', 'phase_death',
            'batting_position_strength', 'overs_completed_normalized',
            'overs_remaining_normalized', 'run_rate_pressure',
            'wicket_pressure', 'time_pressure', 'pressure_index',
            'momentum_score', 'team1_encoded', 'team2_encoded',
            'venue_encoded', 'team1_home_advantage', 'team2_home_advantage',
            'toss_winner_team1', 'toss_decision_bat', 'toss_advantage',
            'weather_encoded', 'batting_friendly_weather',
            'bowling_friendly_weather', 'dew_factor', 'conditions_score',
            'match_progress', 'is_powerplay', 'is_middle_overs',
            'is_death_overs', 'time_criticality', 'score_efficiency',
            'wicket_rate', 'target_difficulty', 'team_form_score',
            'recent_performance'
        ]