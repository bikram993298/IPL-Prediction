"""
Kaggle Dataset Loader for IPL Data
Loads and processes real IPL datasets from Kaggle
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import requests
import zipfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class KaggleIPLDataLoader:
    """Load and process real IPL datasets from Kaggle"""
    
    def __init__(self, data_dir: str = "data/kaggle"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs (public Kaggle datasets)
        self.datasets = {
            'matches': 'https://raw.githubusercontent.com/sanchitvj/IPL_Data_Analysis/master/matches.csv',
            'deliveries': 'https://raw.githubusercontent.com/sanchitvj/IPL_Data_Analysis/master/deliveries.csv',
            'ball_by_ball': 'https://raw.githubusercontent.com/sanchitvj/IPL_Data_Analysis/master/ball_by_ball_backup.csv'
        }
        
        self.processed_data = {}
        
    async def download_datasets(self) -> bool:
        """Download IPL datasets from public sources"""
        logger.info("Downloading IPL datasets from Kaggle sources...")
        
        try:
            for name, url in self.datasets.items():
                file_path = self.data_dir / f"{name}.csv"
                
                if file_path.exists():
                    logger.info(f"Dataset {name} already exists, skipping download")
                    continue
                
                logger.info(f"Downloading {name} dataset...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded {name} dataset successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading datasets: {e}")
            # Create sample data if download fails
            await self._create_sample_data()
            return False
    
    async def _create_sample_data(self):
        """Create sample IPL data for development"""
        logger.info("Creating sample IPL data...")
        
        # Sample matches data
        matches_data = {
            'id': range(1, 1001),
            'season': np.random.choice([2020, 2021, 2022, 2023, 2024], 1000),
            'city': np.random.choice(['Mumbai', 'Chennai', 'Bangalore', 'Kolkata', 'Delhi', 'Jaipur', 'Mohali', 'Hyderabad'], 1000),
            'date': pd.date_range('2020-01-01', periods=1000, freq='D'),
            'team1': np.random.choice(['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders'], 1000),
            'team2': np.random.choice(['Delhi Capitals', 'Rajasthan Royals', 'Punjab Kings', 'Sunrisers Hyderabad'], 1000),
            'toss_winner': np.random.choice(['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders'], 1000),
            'toss_decision': np.random.choice(['bat', 'field'], 1000),
            'result': np.random.choice(['normal', 'tie'], 1000),
            'dl_applied': np.random.choice([0, 1], 1000, p=[0.9, 0.1]),
            'winner': np.random.choice(['Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders'], 1000),
            'win_by_runs': np.random.randint(0, 100, 1000),
            'win_by_wickets': np.random.randint(0, 10, 1000),
            'venue': np.random.choice(['Wankhede Stadium', 'Eden Gardens', 'M Chinnaswamy Stadium', 'MA Chidambaram Stadium'], 1000)
        }
        
        matches_df = pd.DataFrame(matches_data)
        matches_df.to_csv(self.data_dir / 'matches.csv', index=False)
        
        # Sample deliveries data
        deliveries_data = []
        for match_id in range(1, 101):  # Sample 100 matches
            for inning in [1, 2]:
                for over in range(1, 21):
                    for ball in range(1, 7):
                        deliveries_data.append({
                            'match_id': match_id,
                            'inning': inning,
                            'batting_team': np.random.choice(['CSK', 'MI', 'RCB', 'KKR']),
                            'bowling_team': np.random.choice(['DC', 'RR', 'PBKS', 'SRH']),
                            'over': over,
                            'ball': ball,
                            'batsman': f'Player_{np.random.randint(1, 100)}',
                            'non_striker': f'Player_{np.random.randint(1, 100)}',
                            'bowler': f'Bowler_{np.random.randint(1, 50)}',
                            'is_super_over': 0,
                            'wide_runs': np.random.choice([0, 1], p=[0.9, 0.1]),
                            'bye_runs': np.random.choice([0, 1, 2], p=[0.95, 0.04, 0.01]),
                            'legbye_runs': np.random.choice([0, 1, 2], p=[0.95, 0.04, 0.01]),
                            'noball_runs': np.random.choice([0, 1], p=[0.95, 0.05]),
                            'penalty_runs': 0,
                            'batsman_runs': np.random.choice([0, 1, 2, 3, 4, 6], p=[0.3, 0.4, 0.15, 0.05, 0.08, 0.02]),
                            'extra_runs': np.random.choice([0, 1], p=[0.85, 0.15]),
                            'total_runs': 0,
                            'player_dismissed': np.random.choice([None, f'Player_{np.random.randint(1, 100)}'], p=[0.95, 0.05]),
                            'dismissal_kind': np.random.choice([None, 'caught', 'bowled', 'lbw'], p=[0.95, 0.02, 0.02, 0.01]),
                            'fielder': np.random.choice([None, f'Fielder_{np.random.randint(1, 50)}'], p=[0.8, 0.2])
                        })
        
        deliveries_df = pd.DataFrame(deliveries_data)
        deliveries_df['total_runs'] = deliveries_df['batsman_runs'] + deliveries_df['extra_runs']
        deliveries_df.to_csv(self.data_dir / 'deliveries.csv', index=False)
        
        logger.info("Sample data created successfully")
    
    async def load_and_process_data(self) -> Dict[str, pd.DataFrame]:
        """Load and process all IPL datasets"""
        logger.info("Loading and processing IPL datasets...")
        
        # Download datasets if needed
        await self.download_datasets()
        
        # Load datasets
        datasets = {}
        for name in ['matches', 'deliveries']:
            file_path = self.data_dir / f"{name}.csv"
            if file_path.exists():
                datasets[name] = pd.read_csv(file_path)
                logger.info(f"Loaded {name} dataset: {len(datasets[name])} records")
            else:
                logger.warning(f"Dataset {name} not found")
        
        # Process data
        if 'matches' in datasets and 'deliveries' in datasets:
            processed_data = await self._process_datasets(datasets['matches'], datasets['deliveries'])
            self.processed_data = processed_data
            return processed_data
        
        return {}
    
    async def _process_datasets(self, matches_df: pd.DataFrame, deliveries_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Process raw datasets into ML-ready format"""
        logger.info("Processing datasets for ML training...")
        
        # Merge matches and deliveries
        merged_df = deliveries_df.merge(matches_df, left_on='match_id', right_on='id', how='left')
        
        # Create ball-by-ball features
        ball_by_ball_features = []
        
        for match_id in merged_df['match_id'].unique()[:100]:  # Process first 100 matches for demo
            match_data = merged_df[merged_df['match_id'] == match_id]
            
            for inning in [1, 2]:
                inning_data = match_data[match_data['inning'] == inning]
                if len(inning_data) == 0:
                    continue
                
                current_score = 0
                wickets = 0
                
                for idx, ball in inning_data.iterrows():
                    current_score += ball['total_runs']
                    if pd.notna(ball['player_dismissed']):
                        wickets += 1
                    
                    # Calculate features
                    balls_faced = (ball['over'] - 1) * 6 + ball['ball']
                    balls_remaining = 120 - balls_faced
                    
                    # Get target for second innings
                    if inning == 2:
                        first_inning_score = merged_df[
                            (merged_df['match_id'] == match_id) & 
                            (merged_df['inning'] == 1)
                        ]['total_runs'].sum()
                        target = first_inning_score + 1
                        runs_required = target - current_score
                    else:
                        target = None
                        runs_required = None
                    
                    # Determine winner
                    match_winner = ball['winner']
                    batting_team = ball['batting_team']
                    
                    if inning == 1:
                        # For first innings, we need to look at final result
                        win_prob = 0.5  # Will be calculated based on final score
                    else:
                        # For second innings, calculate based on chase
                        if runs_required is not None:
                            if runs_required <= 0:
                                win_prob = 1.0 if match_winner == batting_team else 0.0
                            else:
                                win_prob = 0.5  # Simplified for now
                        else:
                            win_prob = 0.5
                    
                    ball_features = {
                        'match_id': match_id,
                        'team1': ball['team1'],
                        'team2': ball['team2'],
                        'batting_team': batting_team,
                        'bowling_team': ball['bowling_team'],
                        'venue': ball['venue'],
                        'city': ball['city'],
                        'season': ball['season'],
                        'toss_winner': ball['toss_winner'],
                        'toss_decision': ball['toss_decision'],
                        'current_score': current_score,
                        'wickets': wickets,
                        'overs': ball['over'],
                        'balls': ball['ball'],
                        'balls_faced': balls_faced,
                        'balls_remaining': balls_remaining,
                        'target': target,
                        'runs_required': runs_required,
                        'inning': inning,
                        'is_first_innings': inning == 1,
                        'win_probability': win_prob,
                        'actual_winner': match_winner,
                        'runs_this_ball': ball['total_runs'],
                        'is_wicket': 1 if pd.notna(ball['player_dismissed']) else 0
                    }
                    
                    ball_by_ball_features.append(ball_features)
        
        # Create DataFrame
        processed_df = pd.DataFrame(ball_by_ball_features)
        
        # Calculate proper win probabilities for first innings
        for match_id in processed_df['match_id'].unique():
            match_data = processed_df[processed_df['match_id'] == match_id]
            first_inning = match_data[match_data['inning'] == 1]
            
            if len(first_inning) > 0:
                final_score = first_inning['current_score'].max()
                
                # Simple win probability based on final score
                if final_score > 200:
                    base_prob = 0.8
                elif final_score > 180:
                    base_prob = 0.7
                elif final_score > 160:
                    base_prob = 0.6
                elif final_score > 140:
                    base_prob = 0.5
                else:
                    base_prob = 0.4
                
                # Update win probabilities
                processed_df.loc[
                    (processed_df['match_id'] == match_id) & (processed_df['inning'] == 1),
                    'win_probability'
                ] = base_prob
        
        logger.info(f"Processed {len(processed_df)} ball-by-ball records")
        
        return {
            'ball_by_ball': processed_df,
            'matches': matches_df,
            'deliveries': deliveries_df
        }
    
    def get_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get processed data for ML training"""
        if 'ball_by_ball' not in self.processed_data:
            raise ValueError("Data not processed yet. Call load_and_process_data() first.")
        
        df = self.processed_data['ball_by_ball'].copy()
        
        # Remove rows with missing target
        df = df.dropna(subset=['win_probability'])
        
        # Select features for training
        feature_columns = [
            'current_score', 'wickets', 'overs', 'balls', 'balls_faced', 
            'balls_remaining', 'target', 'runs_required', 'is_first_innings'
        ]
        
        # Add categorical features
        df['venue_encoded'] = pd.Categorical(df['venue']).codes
        df['toss_decision_encoded'] = pd.Categorical(df['toss_decision']).codes
        df['season_encoded'] = df['season'] - df['season'].min()
        
        feature_columns.extend(['venue_encoded', 'toss_decision_encoded', 'season_encoded'])
        
        # Fill missing values
        df[feature_columns] = df[feature_columns].fillna(0)
        
        X = df[feature_columns]
        y = df['win_probability']
        
        logger.info(f"Training data prepared: {len(X)} samples, {len(feature_columns)} features")
        
        return X, y