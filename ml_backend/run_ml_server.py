#!/usr/bin/env python3
"""
Run the ML Backend Server
Development script to start the FastAPI server
"""

import os
import sys
import uvicorn
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to run the ML server"""
    print("=" * 60)
    print("IPL ML PREDICTION SERVER")
    print("=" * 60)
    print("Starting FastAPI server with ML models...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()