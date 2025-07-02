#!/usr/bin/env python3
"""
Run the Advanced ML Backend Server
Production-grade ML server with Kaggle datasets
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 
        'lightgbm', 'catboost', 'tensorflow', 'torch', 
        'optuna', 'fastapi', 'uvicorn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    return True

def main():
    """Main function to run the advanced ML server"""
    print("=" * 80)
    print("🏏 ADVANCED IPL ML PREDICTION SERVER")
    print("=" * 80)
    print("🚀 Features:")
    print("   📊 Real Kaggle IPL Datasets")
    print("   🤖 Ensemble Models: Random Forest, XGBoost, LightGBM, CatBoost")
    print("   🧠 Deep Learning: TensorFlow & PyTorch")
    print("   ⚡ Hyperparameter Optimization")
    print("   🎯 Sub-100ms Predictions")
    print("=" * 80)
    
    # Change to ml_backend directory
    ml_backend_dir = Path(__file__).parent
    os.chdir(ml_backend_dir)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return
    
    print("🔧 Starting advanced ML server...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("📈 Training Status: http://localhost:8000/training-status")
    print("=" * 80)
    
    # Run the server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "enhanced_main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()