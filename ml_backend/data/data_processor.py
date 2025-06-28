"""
Advanced Data Processing for IPL ML Models
Handles data generation, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class AdvancedDataProcessor:
    """Advanced data processing for IPL predictions"""
    
    def __init__(self):
        self.teams = [
            'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Capitals', 'Rajasthan Royals',
            'Punjab Kings', 'Sunrisers Hyderabad'
        ]
        
        self.venues = [
            'Wankhede Stadium, Mumbai', 'Eden Gardens, Kolkata',
            'M. Chinnaswamy Stadium, Bangalore', 'MA Chidambaram Stadium, Chennai',
            'Arun Jaitley Stadium, Delhi', 'Sawai Mansingh Stadium, Jaipur',
            'PCA Stadium, Mohali', 'Rajiv Gandhi International Stadium, Hyderabad'
        ]
        
        self.weather_conditions = ['Clear', 'Partly Cloudy', 'Overcast', 'Light Rain', 'Dew Expected']
    
    async def generate_training_data(self, size: int = 10000) -> pd.DataFrame:
        """Generate comprehensive training dataset"""
        logger.info(f"Generating {size} training samples...")
        
        matches = []
        
        for match_id in range(size // 100):  # Each match has ~100 ball-by-ball entries
            match_data = await self._generate_single_match(match_id)
            matches.extend(match_data)
        
        df = pd.DataFrame(matches)
        logger.info(f"Generated {len(df)} training samples")
        return df
    
    async def _generate_single_match(self, match_id: int) -> List[Dict[str, Any]]:
        """Generate realistic ball-by-ball data for a single match"""
        # Match setup
        team1, team2 = random.sample(self.teams, 2)
        venue = random.choice(self.venues)
        weather = random.choice(self.weather_conditions)
        
        # Toss
        toss_winner = random.choice([team1, team2])
        toss_decision = random.choice(['bat', 'bowl'])
        
        # First innings score
        first_innings_score = self._generate_realistic_score()
        
        # Generate chase data
        match_data = []
        current_score = 0
        wickets = 0
        
        # Determine batting/bowling teams for second innings
        if (toss_winner == team1 and toss_decision == 'bat') or (toss_winner == team2 and toss_decision == 'bowl'):
            batting_team = team1
            bowling_team = team2
        else:
            batting_team = team2
            bowling_team = team1
        
        for over in range(20):
            for ball in range(6):
                if wickets >= 10 or current_score >= first_innings_score:
                    break
                
                balls_faced = over * 6 + ball + 1
                balls_remaining = 120 - balls_faced
                runs_required = first_innings_score - current_score
                
                # Generate ball outcome
                runs_this_ball = self._generate_ball_outcome(balls_remaining, runs_required, wickets)
                is_wicket = self._should_wicket_fall(balls_remaining, runs_required, wickets)
                
                if is_wicket and wickets < 10:
                    wickets += 1
                
                current_score += runs_this_ball
                
                # Calculate win probability
                win_prob = self._calculate_win_probability(
                    current_score, wickets, balls_remaining, first_innings_score
                )
                
                # Store ball data
                ball_data = {
                    'match_id': match_id,
                    'team1': team1,
                    'team2': team2,
                    'batting_team': batting_team,
                    'bowling_team': bowling_team,
                    'venue': venue,
                    'weather': weather,
                    'toss_winner': toss_winner,
                    'toss_decision': toss_decision,
                    'target': first_innings_score,
                    'current_score': current_score,
                    'wickets': wickets,
                    'overs': over,
                    'balls': ball,
                    'balls_remaining': balls_remaining,
                    'runs_required': runs_required,
                    'runs_this_ball': runs_this_ball,
                    'is_wicket': is_wicket,
                    'win_probability': win_prob,
                    'is_first_innings': False
                }
                
                match_data.append(ball_data)
                
                if current_score >= first_innings_score:
                    break
            
            if wickets >= 10 or current_score >= first_innings_score:
                break
        
        return match_data
    
    def _generate_realistic_score(self) -> int:
        """Generate realistic T20 score"""
        # Base score with normal distribution
        base_score = np.random.normal(160, 25)
        
        # Add some variance for different conditions
        variance = np.random.normal(0, 15)
        
        final_score = int(base_score + variance)
        return max(80, min(250, final_score))
    
    def _generate_ball_outcome(self, balls_remaining: int, runs_required: int, wickets: int) -> int:
        """Generate realistic runs for a ball based on match situation"""
        if balls_remaining <= 0:
            return 0
        
        required_rate = (runs_required / balls_remaining) * 6 if balls_remaining > 0 else 0
        
        # Adjust probabilities based on required rate and pressure
        if required_rate > 12:  # Very high pressure
            outcomes = [0, 0, 1, 2, 4, 6]
            weights = [0.35, 0.15, 0.20, 0.10, 0.15, 0.05]
        elif required_rate > 9:  # High pressure
            outcomes = [0, 1, 1, 2, 4, 6]
            weights = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
        elif required_rate > 6:  # Medium pressure
            outcomes = [0, 1, 1, 2, 3, 4, 6]
            weights = [0.20, 0.30, 0.20, 0.15, 0.05, 0.08, 0.02]
        else:  # Low pressure
            outcomes = [0, 1, 1, 2, 3, 4]
            weights = [0.15, 0.40, 0.25, 0.15, 0.03, 0.02]
        
        # Adjust for wickets lost (more conservative with fewer wickets)
        if wickets >= 7:
            # More conservative
            weights = [w * 1.2 if o <= 1 else w * 0.8 for w, o in zip(weights, outcomes)]
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.random.choice(outcomes, p=weights)
    
    def _should_wicket_fall(self, balls_remaining: int, runs_required: int, wickets: int) -> bool:
        """Determine if a wicket should fall"""
        base_prob = 0.05  # 5% base probability
        
        # Increase under pressure
        if balls_remaining > 0:
            required_rate = (runs_required / balls_remaining) * 6
            if required_rate > 10:
                base_prob += 0.03
            elif required_rate > 8:
                base_prob += 0.02
        
        # Decrease if many wickets already fallen
        if wickets >= 7:
            base_prob *= 0.6
        elif wickets >= 5:
            base_prob *= 0.8
        
        return np.random.random() < base_prob
    
    def _calculate_win_probability(self, current_score: int, wickets: int, 
                                 balls_remaining: int, target: int) -> float:
        """Calculate realistic win probability"""
        if current_score >= target:
            return 0.95  # Almost certain win
        
        if wickets >= 10 or balls_remaining <= 0:
            return 0.05  # Almost certain loss
        
        runs_required = target - current_score
        required_rate = (runs_required / balls_remaining) * 6 if balls_remaining > 0 else float('inf')
        
        # Base probability based on required rate
        if required_rate <= 4:
            base_prob = 0.85
        elif required_rate <= 6:
            base_prob = 0.75
        elif required_rate <= 8:
            base_prob = 0.55
        elif required_rate <= 10:
            base_prob = 0.35
        elif required_rate <= 12:
            base_prob = 0.20
        else:
            base_prob = 0.10
        
        # Adjust for wickets in hand
        wickets_factor = (10 - wickets) / 10
        adjusted_prob = base_prob * (0.3 + 0.7 * wickets_factor)
        
        # Adjust for balls remaining
        if balls_remaining < 12:  # Last 2 overs
            adjusted_prob *= 0.85
        elif balls_remaining < 24:  # Last 4 overs
            adjusted_prob *= 0.92
        
        # Add some randomness
        noise = np.random.normal(0, 0.05)
        final_prob = adjusted_prob + noise
        
        return max(0.05, min(0.95, final_prob))
    
    async def process_match_state(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate match state data"""
        processed = match_data.copy()
        
        # Ensure numeric fields are properly typed
        numeric_fields = ['current_score', 'wickets', 'overs', 'balls', 'target']
        for field in numeric_fields:
            if field in processed:
                processed[field] = float(processed[field])
        
        # Add derived fields
        total_balls = processed.get('overs', 0) * 6 + processed.get('balls', 0)
        processed['total_balls_faced'] = total_balls
        processed['balls_remaining'] = 120 - total_balls
        
        if processed.get('target', 0) > 0:
            processed['runs_required'] = processed['target'] - processed['current_score']
        else:
            processed['runs_required'] = 0
        
        return processed
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate dataset"""
        logger.info("Cleaning dataset...")
        
        # Remove invalid rows
        df = df.dropna(subset=['current_score', 'wickets', 'overs'])
        
        # Ensure valid ranges
        df = df[df['current_score'] >= 0]
        df = df[df['wickets'] >= 0]
        df = df[df['wickets'] <= 10]
        df = df[df['overs'] >= 0]
        df = df[df['overs'] <= 20]
        df = df[df['balls'] >= 0]
        df = df[df['balls'] <= 5]
        
        # Ensure win probability is in valid range
        if 'win_probability' in df.columns:
            df = df[df['win_probability'] >= 0]
            df = df[df['win_probability'] <= 1]
        
        logger.info(f"Cleaned dataset: {len(df)} samples remaining")
        return df
    
    def add_noise_for_robustness(self, df: pd.DataFrame, noise_level: float = 0.02) -> pd.DataFrame:
        """Add small amount of noise to make model more robust"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col not in ['match_id', 'wickets', 'overs', 'balls']:  # Don't add noise to discrete values
                noise = np.random.normal(0, df[col].std() * noise_level, len(df))
                df[col] = df[col] + noise
        
        return df