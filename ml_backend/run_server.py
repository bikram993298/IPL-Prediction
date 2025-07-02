#!/usr/bin/env python3
"""
Run the ML Backend Server
Simple script to start the FastAPI server
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Main function to run the ML server"""
    print("=" * 60)
    print("🏏 IPL ML PREDICTION SERVER")
    print("=" * 60)
    print("🚀 Starting FastAPI server with ML models...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("=" * 60)
    
    # Change to ml_backend directory
    ml_backend_dir = Path(__file__).parent
    os.chdir(ml_backend_dir)
    
    # Run the server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "simplified_main:app", 
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