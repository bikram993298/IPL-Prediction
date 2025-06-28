import React, { useState, useEffect } from 'react';
import { Zap, TrendingUp, AlertCircle, CheckCircle, XCircle, RefreshCw } from 'lucide-react';
import LiveMatchSelector from './LiveMatchSelector';
import { cricketApi } from '../services/cricketApi';
import type { LiveMatch } from '../services/cricketApi';
import type { Team } from './TeamSelector';

interface LiveMatchIntegrationProps {
  onMatchDataUpdate: (data: {
    team1: Team | null;
    team2: Team | null;
    currentScore: number;
    wickets: number;
    overs: number;
    balls: number;
    target: number;
    venue: string;
    tossWinner: 'team1' | 'team2' | null;
    tossDecision: 'bat' | 'bowl' | null;
    isFirstInnings: boolean;
  }) => void;
}

// Team mapping for consistent data
const teamMapping: Record<string, Team> = {
  'Chennai Super Kings': { id: 'csk', name: 'Chennai Super Kings', shortName: 'CSK', colors: ['from-yellow-400', 'to-yellow-600'], primaryColor: 'bg-yellow-500' },
  'Mumbai Indians': { id: 'mi', name: 'Mumbai Indians', shortName: 'MI', colors: ['from-blue-500', 'to-blue-700'], primaryColor: 'bg-blue-600' },
  'Royal Challengers Bangalore': { id: 'rcb', name: 'Royal Challengers Bangalore', shortName: 'RCB', colors: ['from-red-500', 'to-red-700'], primaryColor: 'bg-red-600' },
  'Kolkata Knight Riders': { id: 'kkr', name: 'Kolkata Knight Riders', shortName: 'KKR', colors: ['from-purple-500', 'to-purple-700'], primaryColor: 'bg-purple-600' },
  'Delhi Capitals': { id: 'dc', name: 'Delhi Capitals', shortName: 'DC', colors: ['from-blue-400', 'to-blue-600'], primaryColor: 'bg-blue-500' },
  'Rajasthan Royals': { id: 'rr', name: 'Rajasthan Royals', shortName: 'RR', colors: ['from-pink-500', 'to-pink-700'], primaryColor: 'bg-pink-600' },
  'Punjab Kings': { id: 'pbks', name: 'Punjab Kings', shortName: 'PBKS', colors: ['from-red-400', 'to-red-600'], primaryColor: 'bg-red-500' },
  'Sunrisers Hyderabad': { id: 'srh', name: 'Sunrisers Hyderabad', shortName: 'SRH', colors: ['from-orange-500', 'to-orange-700'], primaryColor: 'bg-orange-600' },
};

export default function LiveMatchIntegration({ onMatchDataUpdate }: LiveMatchIntegrationProps) {
  const [selectedMatch, setSelectedMatch] = useState<LiveMatch | null>(null);
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [apiStatus, setApiStatus] = useState<{ success: boolean; remaining?: number; message: string } | null>(null);
  const [checkingApi, setCheckingApi] = useState(false);

  // Check API status on component mount
  useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    setCheckingApi(true);
    try {
      const status = await cricketApi.checkApiStatus();
      setApiStatus(status);
    } catch (error) {
      setApiStatus({
        success: false,
        message: 'Failed to check API status'
      });
    } finally {
      setCheckingApi(false);
    }
  };

  const handleMatchSelect = (match: LiveMatch) => {
    setSelectedMatch(match);
    setIsLiveMode(true);
    
    // Map live match data to app format
    const team1 = teamMapping[match.team1] || null;
    const team2 = teamMapping[match.team2] || null;
    
    onMatchDataUpdate({
      team1,
      team2,
      currentScore: match.currentScore,
      wickets: match.wickets,
      overs: match.overs,
      balls: match.balls,
      target: match.target || 0,
      venue: match.venue,
      tossWinner: match.tossWinner === match.team1 ? 'team1' : 'team2',
      tossDecision: match.tossDecision || null,
      isFirstInnings: !match.target
    });
  };

  const handleDisableLiveMode = () => {
    setIsLiveMode(false);
    setSelectedMatch(null);
  };

  return (
    <div className="space-y-6">
      {/* API Status */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <Zap className="w-5 h-5 text-blue-600" />
            <h3 className="text-lg font-bold text-gray-800">API Status</h3>
          </div>
          <button
            onClick={checkApiStatus}
            disabled={checkingApi}
            className="p-2 rounded-lg bg-blue-50 hover:bg-blue-100 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 text-blue-600 ${checkingApi ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {apiStatus && (
          <div className={`flex items-center space-x-2 p-3 rounded-lg ${
            apiStatus.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
          }`}>
            {apiStatus.success ? (
              <CheckCircle className="w-5 h-5 text-green-600" />
            ) : (
              <XCircle className="w-5 h-5 text-red-600" />
            )}
            <div>
              <p className={`text-sm font-medium ${apiStatus.success ? 'text-green-800' : 'text-red-800'}`}>
                {apiStatus.success ? 'API Connected' : 'API Error'}
              </p>
              <p className={`text-xs ${apiStatus.success ? 'text-green-600' : 'text-red-600'}`}>
                {apiStatus.message}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Live Mode Toggle */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-gray-800">Live Match Integration</h3>
              <p className="text-sm text-gray-600">
                {isLiveMode ? 'Connected to live match data' : 'Connect to real-time cricket scores'}
              </p>
            </div>
          </div>
          
          {isLiveMode && (
            <button
              onClick={handleDisableLiveMode}
              className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
            >
              Disable Live Mode
            </button>
          )}
        </div>

        {isLiveMode && selectedMatch && (
          <div className="mt-4 p-4 bg-white rounded-lg border border-green-200">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-green-700">LIVE DATA ACTIVE</span>
            </div>
            <div className="text-sm text-gray-600">
              <strong>{selectedMatch.team1}</strong> vs <strong>{selectedMatch.team2}</strong>
              <br />
              Score: {selectedMatch.currentScore}/{selectedMatch.wickets} ({selectedMatch.overs}.{selectedMatch.balls} ov)
              {selectedMatch.target && <span> | Target: {selectedMatch.target}</span>}
            </div>
          </div>
        )}
      </div>

      {/* Live Match Selector */}
      {!isLiveMode && (
        <LiveMatchSelector
          onMatchSelect={handleMatchSelect}
          selectedMatch={selectedMatch}
        />
      )}

      {/* API Integration Info */}
      <div className="bg-blue-50 rounded-xl p-6">
        <div className="flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
          <div>
            <h4 className="font-semibold text-blue-800 mb-2">Live Data Integration</h4>
            <div className="text-sm text-blue-700 space-y-2">
              <p><strong>✅ API Key Configured:</strong> 77801300-cbed-4c96-869a-81e31ebc1484</p>
              <p><strong>🔗 Data Source:</strong> CricAPI (cricapi.com)</p>
              <p><strong>📊 Features:</strong></p>
              <ul className="list-disc list-inside ml-4 space-y-1">
                <li>Real-time IPL match scores</li>
                <li>Live ball-by-ball updates</li>
                <li>Match status and venue information</li>
                <li>Toss details and team information</li>
                <li>Automatic 30-second refresh during live matches</li>
              </ul>
              
              <p className="mt-3"><strong>🎯 How it works:</strong></p>
              <ul className="list-disc list-inside ml-4 space-y-1">
                <li>Fetches current IPL matches from CricAPI</li>
                <li>Filters for T20 matches and IPL teams</li>
                <li>Auto-populates all prediction inputs</li>
                <li>Updates win probabilities in real-time</li>
                <li>Falls back to mock data when no live matches</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <TrendingUp className="w-8 h-8 text-green-500 mb-2" />
          <h5 className="font-semibold text-gray-800">Real-time Updates</h5>
          <p className="text-sm text-gray-600">Live score updates every 30 seconds during matches</p>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <Zap className="w-8 h-8 text-blue-500 mb-2" />
          <h5 className="font-semibold text-gray-800">Auto Sync</h5>
          <p className="text-sm text-gray-600">Automatically updates all prediction inputs</p>
        </div>
        
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <AlertCircle className="w-8 h-8 text-purple-500 mb-2" />
          <h5 className="font-semibold text-gray-800">Smart Fallback</h5>
          <p className="text-sm text-gray-600">Mock data when no live IPL matches available</p>
        </div>
      </div>
    </div>
  );
}