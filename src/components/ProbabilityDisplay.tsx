import React from 'react';
import { TrendingUp, Award, AlertCircle } from 'lucide-react';
import type { Team } from './TeamSelector';

interface ProbabilityDisplayProps {
  team1: Team | null;
  team2: Team | null;
  probability: {
    team1: number;
    team2: number;
    confidence: 'low' | 'medium' | 'high';
  };
}

export default function ProbabilityDisplay({ team1, team2, probability }: ProbabilityDisplayProps) {
  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high': return 'text-green-600';
      case 'medium': return 'text-yellow-600';
      case 'low': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getConfidenceIcon = (confidence: string) => {
    switch (confidence) {
      case 'high': return Award;
      case 'medium': return TrendingUp;
      case 'low': return AlertCircle;
      default: return AlertCircle;
    }
  };

  const ConfidenceIcon = getConfidenceIcon(probability.confidence);

  return (
    <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-gray-800">Win Probability</h3>
        <div className={`flex items-center space-x-1 ${getConfidenceColor(probability.confidence)}`}>
          <ConfidenceIcon className="w-5 h-5" />
          <span className="text-sm font-semibold capitalize">{probability.confidence} Confidence</span>
        </div>
      </div>

      <div className="space-y-6">
        {/* Team 1 Probability */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              {team1 && (
                <div className={`w-8 h-8 rounded-full flex items-center justify-center`}
                     style={{ backgroundColor: team1.primaryColor.replace('bg-', '#') }}>
                  <span className="text-white text-xs font-bold">{team1.shortName}</span>
                </div>
              )}
              <span className="font-semibold text-gray-800">
                {team1?.name || 'Team 1'}
              </span>
            </div>
            <span className="text-2xl font-bold text-gray-800">{probability.team1.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
            <div 
              className={`h-full transition-all duration-1000 ease-out ${
                team1 ? `bg-gradient-to-r ${team1.colors.join(' ')}` : 'bg-blue-500'
              }`}
              style={{ width: `${probability.team1}%` }}
            ></div>
          </div>
        </div>

        {/* Team 2 Probability */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              {team2 && (
                <div className={`w-8 h-8 rounded-full flex items-center justify-center`}
                     style={{ backgroundColor: team2.primaryColor.replace('bg-', '#') }}>
                  <span className="text-white text-xs font-bold">{team2.shortName}</span>
                </div>
              )}
              <span className="font-semibold text-gray-800">
                {team2?.name || 'Team 2'}
              </span>
            </div>
            <span className="text-2xl font-bold text-gray-800">{probability.team2.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
            <div 
              className={`h-full transition-all duration-1000 ease-out ${
                team2 ? `bg-gradient-to-r ${team2.colors.join(' ')}` : 'bg-red-500'
              }`}
              style={{ width: `${probability.team2}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Winner Prediction */}
      <div className="mt-6 p-4 bg-white rounded-lg border border-gray-200">
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1">Predicted Winner</div>
          <div className="text-lg font-bold text-gray-800">
            {probability.team1 > probability.team2 
              ? (team1?.name || 'Team 1')
              : (team2?.name || 'Team 2')
            }
          </div>
          <div className="text-sm text-gray-500">
            {Math.max(probability.team1, probability.team2).toFixed(1)}% chance to win
          </div>
        </div>
      </div>
    </div>
  );
}