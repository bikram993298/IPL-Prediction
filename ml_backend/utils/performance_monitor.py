"""
Performance Monitoring and Analytics
Tracks model performance, predictions, and system metrics
"""

import time
import psutil
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import pandas as pd
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and track ML model performance"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.predictions_history = deque(maxlen=max_history)
        self.performance_metrics = defaultdict(list)
        self.system_metrics = defaultdict(list)
        self.start_time = datetime.now()
        
    async def log_prediction(self, input_data: Dict[str, Any], prediction_result: Dict[str, Any]):
        """Log a prediction for monitoring"""
        prediction_log = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data,
            'output': prediction_result,
            'processing_time': prediction_result.get('processing_time', 0)
        }
        
        self.predictions_history.append(prediction_log)
        
        # Update metrics
        self.performance_metrics['total_predictions'].append(len(self.predictions_history))
        self.performance_metrics['avg_processing_time'].append(
            prediction_result.get('processing_time', 0)
        )
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.predictions_history:
            return {
                'status': 'no_data',
                'message': 'No predictions logged yet'
            }
        
        # Calculate metrics
        recent_predictions = list(self.predictions_history)[-100:]  # Last 100 predictions
        
        processing_times = [p.get('output', {}).get('processing_time', 0) for p in recent_predictions]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Confidence distribution
        confidences = [p.get('output', {}).get('confidence', 'unknown') for p in recent_predictions]
        confidence_dist = {
            'high': confidences.count('high'),
            'medium': confidences.count('medium'),
            'low': confidences.count('low')
        }
        
        # System metrics
        system_info = self.get_system_metrics()
        
        return {
            'status': 'active',
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'total_predictions': len(self.predictions_history),
            'recent_predictions_count': len(recent_predictions),
            'average_processing_time_ms': avg_processing_time * 1000,
            'confidence_distribution': confidence_dist,
            'system_metrics': system_info,
            'last_prediction_time': recent_predictions[-1]['timestamp'] if recent_predictions else None
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {'error': 'Failed to retrieve system metrics'}
    
    def get_memory_usage(self) -> str:
        """Get current memory usage"""
        try:
            memory = psutil.virtual_memory()
            return f"{memory.percent:.1f}%"
        except:
            return "unknown"
    
    async def get_prediction_analytics(self) -> Dict[str, Any]:
        """Get detailed prediction analytics"""
        if not self.predictions_history:
            return {'status': 'no_data'}
        
        predictions = list(self.predictions_history)
        
        # Time-based analysis
        hourly_counts = defaultdict(int)
        for pred in predictions:
            hour = datetime.fromisoformat(pred['timestamp']).hour
            hourly_counts[hour] += 1
        
        # Team analysis
        team_predictions = defaultdict(int)
        for pred in predictions:
            team1 = pred.get('input', {}).get('team1', 'unknown')
            team2 = pred.get('input', {}).get('team2', 'unknown')
            team_predictions[team1] += 1
            team_predictions[team2] += 1
        
        # Venue analysis
        venue_predictions = defaultdict(int)
        for pred in predictions:
            venue = pred.get('input', {}).get('venue', 'unknown')
            venue_predictions[venue] += 1
        
        return {
            'total_predictions': len(predictions),
            'hourly_distribution': dict(hourly_counts),
            'team_frequency': dict(team_predictions),
            'venue_frequency': dict(venue_predictions),
            'date_range': {
                'start': predictions[0]['timestamp'] if predictions else None,
                'end': predictions[-1]['timestamp'] if predictions else None
            }
        }
    
    async def export_logs(self, filepath: str):
        """Export prediction logs to file"""
        try:
            logs_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_predictions': len(self.predictions_history),
                'predictions': list(self.predictions_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(logs_data, f, indent=2)
            
            logger.info(f"Exported {len(self.predictions_history)} prediction logs to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export logs: {e}")
            raise
    
    def clear_old_logs(self, days_to_keep: int = 7):
        """Clear old prediction logs"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        original_count = len(self.predictions_history)
        
        # Filter predictions
        filtered_predictions = [
            pred for pred in self.predictions_history
            if datetime.fromisoformat(pred['timestamp']) > cutoff_date
        ]
        
        self.predictions_history.clear()
        self.predictions_history.extend(filtered_predictions)
        
        removed_count = original_count - len(self.predictions_history)
        logger.info(f"Cleared {removed_count} old prediction logs")
    
    async def get_model_accuracy_metrics(self) -> Dict[str, Any]:
        """Calculate model accuracy metrics (would need actual outcomes)"""
        # This would require actual match outcomes to calculate real accuracy
        # For now, return placeholder metrics
        
        return {
            'note': 'Accuracy metrics require actual match outcomes',
            'predictions_made': len(self.predictions_history),
            'confidence_breakdown': await self._get_confidence_breakdown(),
            'prediction_distribution': await self._get_prediction_distribution()
        }
    
    async def _get_confidence_breakdown(self) -> Dict[str, int]:
        """Get breakdown of prediction confidence levels"""
        confidence_counts = defaultdict(int)
        
        for pred in self.predictions_history:
            confidence = pred.get('output', {}).get('confidence', 'unknown')
            confidence_counts[confidence] += 1
        
        return dict(confidence_counts)
    
    async def _get_prediction_distribution(self) -> Dict[str, Any]:
        """Get distribution of prediction probabilities"""
        probabilities = []
        
        for pred in self.predictions_history:
            team1_prob = pred.get('output', {}).get('team1_probability', 0.5)
            probabilities.append(team1_prob)
        
        if not probabilities:
            return {'status': 'no_data'}
        
        probabilities = pd.Series(probabilities)
        
        return {
            'mean': float(probabilities.mean()),
            'std': float(probabilities.std()),
            'min': float(probabilities.min()),
            'max': float(probabilities.max()),
            'quartiles': {
                '25%': float(probabilities.quantile(0.25)),
                '50%': float(probabilities.quantile(0.50)),
                '75%': float(probabilities.quantile(0.75))
            }
        }