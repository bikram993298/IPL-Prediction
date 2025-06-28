import React, { useState, useEffect } from 'react';
import { Server, Wifi, WifiOff, RefreshCw, BarChart3, Brain } from 'lucide-react';
import { mlApi } from '../services/mlApi';

export default function MLBackendStatus() {
  const [isConnected, setIsConnected] = useState(false);
  const [checking, setChecking] = useState(false);
  const [performance, setPerformance] = useState<any>(null);

  const checkConnection = async () => {
    setChecking(true);
    try {
      const connected = await mlApi.checkConnection();
      setIsConnected(connected);
      
      if (connected) {
        const perfData = await mlApi.getModelPerformance();
        setPerformance(perfData);
      }
    } catch (error) {
      setIsConnected(false);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    checkConnection();
    
    // Check every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-purple-600" />
          <h3 className="text-lg font-bold text-gray-800">ML Backend Status</h3>
        </div>
        
        <button
          onClick={checkConnection}
          disabled={checking}
          className="p-2 rounded-lg bg-purple-50 hover:bg-purple-100 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 text-purple-600 ${checking ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className={`flex items-center space-x-3 p-4 rounded-lg border-2 ${
        isConnected 
          ? 'border-green-200 bg-green-50' 
          : 'border-red-200 bg-red-50'
      }`}>
        {isConnected ? (
          <Wifi className="w-6 h-6 text-green-600" />
        ) : (
          <WifiOff className="w-6 h-6 text-red-600" />
        )}
        
        <div>
          <div className={`font-semibold ${isConnected ? 'text-green-800' : 'text-red-800'}`}>
            {isConnected ? 'ML Backend Connected' : 'ML Backend Offline'}
          </div>
          <div className={`text-sm ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
            {isConnected 
              ? 'Advanced ML predictions active' 
              : 'Using fallback prediction algorithm'
            }
          </div>
        </div>
      </div>

      {isConnected && performance && (
        <div className="mt-4 space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="text-sm text-blue-600">Total Predictions</div>
              <div className="text-xl font-bold text-blue-800">
                {performance.total_predictions || 0}
              </div>
            </div>
            
            <div className="bg-green-50 p-3 rounded-lg">
              <div className="text-sm text-green-600">Avg Response Time</div>
              <div className="text-xl font-bold text-green-800">
                {performance.average_processing_time_ms?.toFixed(1) || 0}ms
              </div>
            </div>
          </div>

          <div className="bg-purple-50 p-3 rounded-lg">
            <div className="text-sm text-purple-600 mb-2">Confidence Distribution</div>
            <div className="flex space-x-4 text-sm">
              <span className="text-green-700">
                High: {performance.confidence_distribution?.high || 0}
              </span>
              <span className="text-yellow-700">
                Medium: {performance.confidence_distribution?.medium || 0}
              </span>
              <span className="text-red-700">
                Low: {performance.confidence_distribution?.low || 0}
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <div className="flex items-center space-x-2 mb-2">
          <Server className="w-4 h-4 text-gray-600" />
          <span className="text-sm font-medium text-gray-700">Backend Features</span>
        </div>
        <div className="text-xs text-gray-600 space-y-1">
          <div>• Ensemble ML Models (RF, XGBoost, LightGBM, CatBoost)</div>
          <div>• Deep Learning (TensorFlow & PyTorch)</div>
          <div>• Advanced Feature Engineering</div>
          <div>• Real-time Performance Monitoring</div>
          <div>• Model Training & Management</div>
        </div>
      </div>
    </div>
  );
}