import React from 'react';
import { RefreshCw, Wifi, WifiOff, Clock } from 'lucide-react';
import { useLiveMatches } from '../hooks/useLiveMatches';
import type { LiveMatch } from '../services/cricketApi';

interface LiveMatchSelectorProps {
  onMatchSelect: (match: LiveMatch) => void;
  selectedMatch: LiveMatch | null;
}

export default function LiveMatchSelector({ onMatchSelect, selectedMatch }: LiveMatchSelectorProps) {
  const { matches, loading, error, lastUpdated, refetch } = useLiveMatches();

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'live': return 'text-green-600 bg-green-50';
      case 'completed': return 'text-gray-600 bg-gray-50';
      case 'upcoming': return 'text-blue-600 bg-blue-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'live': return <Wifi className="w-4 h-4" />;
      case 'completed': return <WifiOff className="w-4 h-4" />;
      case 'upcoming': return <Clock className="w-4 h-4" />;
      default: return <WifiOff className="w-4 h-4" />;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Wifi className="w-5 h-5 text-green-600" />
          <h3 className="text-lg font-bold text-gray-800">Live Matches</h3>
        </div>
        
        <div className="flex items-center space-x-3">
          {lastUpdated && (
            <span className="text-xs text-gray-500">
              Updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={refetch}
            disabled={loading}
            className="p-2 rounded-lg bg-blue-50 hover:bg-blue-100 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 text-blue-600 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {loading && matches.length === 0 ? (
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="w-6 h-6 animate-spin text-blue-600" />
          <span className="ml-2 text-gray-600">Loading live matches...</span>
        </div>
      ) : (
        <div className="space-y-3">
          {matches.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <WifiOff className="w-8 h-8 mx-auto mb-2" />
              <p>No live matches available</p>
            </div>
          ) : (
            matches.map((match) => (
              <div
                key={match.matchId}
                onClick={() => onMatchSelect(match)}
                className={`p-4 border-2 rounded-lg cursor-pointer transition-all hover:shadow-md ${
                  selectedMatch?.matchId === match.matchId
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${getStatusColor(match.status)}`}>
                      {getStatusIcon(match.status)}
                      <span className="capitalize">{match.status}</span>
                    </span>
                  </div>
                  
                  {match.status === 'live' && (
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                      <span className="text-xs text-red-600 font-medium">LIVE</span>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-gray-800">{match.team1}</span>
                    <span className="text-sm text-gray-600">vs</span>
                    <span className="font-semibold text-gray-800">{match.team2}</span>
                  </div>

                  {match.status === 'live' && (
                    <div className="flex items-center justify-between text-sm">
                      <div className="text-gray-600">
                        <span className="font-medium">{match.currentScore}/{match.wickets}</span>
                        <span className="ml-2">({match.overs}.{match.balls} ov)</span>
                      </div>
                      
                      {match.target && (
                        <div className="text-gray-600">
                          Target: <span className="font-medium">{match.target}</span>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="text-xs text-gray-500">
                    {match.venue}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <p className="text-xs text-gray-600">
          <strong>Data Sources:</strong> CricAPI, RapidAPI Cricket Live Data
        </p>
        <p className="text-xs text-gray-500 mt-1">
          Real-time data updates every 30 seconds during live matches
        </p>
      </div>
    </div>
  );
}