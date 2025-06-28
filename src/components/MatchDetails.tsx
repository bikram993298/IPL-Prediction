import React from 'react';
import { MapPin, Calendar, Cloud, Users } from 'lucide-react';

interface MatchDetailsProps {
  venue: string;
  onVenueChange: (venue: string) => void;
  tossWinner: 'team1' | 'team2' | null;
  onTossWinnerChange: (winner: 'team1' | 'team2' | null) => void;
  tossDecision: 'bat' | 'bowl' | null;
  onTossDecisionChange: (decision: 'bat' | 'bowl' | null) => void;
  weather: string;
  onWeatherChange: (weather: string) => void;
  team1Name: string;
  team2Name: string;
}

const venues = [
  'Wankhede Stadium, Mumbai',
  'Eden Gardens, Kolkata',
  'M. Chinnaswamy Stadium, Bangalore',
  'MA Chidambaram Stadium, Chennai',
  'Arun Jaitley Stadium, Delhi',
  'Sawai Mansingh Stadium, Jaipur',
  'PCA Stadium, Mohali',
  'Rajiv Gandhi International Stadium, Hyderabad',
];

const weatherConditions = ['Clear', 'Partly Cloudy', 'Overcast', 'Light Rain', 'Dew Expected'];

export default function MatchDetails({
  venue,
  onVenueChange,
  tossWinner,
  onTossWinnerChange,
  tossDecision,
  onTossDecisionChange,
  weather,
  onWeatherChange,
  team1Name,
  team2Name,
}: MatchDetailsProps) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 space-y-6">
      <div className="flex items-center space-x-2 mb-4">
        <Calendar className="w-5 h-5 text-blue-600" />
        <h3 className="text-lg font-bold text-gray-800">Match Details</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-2">
          <label className="flex items-center space-x-2 text-sm font-semibold text-gray-700">
            <MapPin className="w-4 h-4" />
            <span>Venue</span>
          </label>
          <select
            value={venue}
            onChange={(e) => onVenueChange(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select Venue</option>
            {venues.map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="flex items-center space-x-2 text-sm font-semibold text-gray-700">
            <Cloud className="w-4 h-4" />
            <span>Weather</span>
          </label>
          <select
            value={weather}
            onChange={(e) => onWeatherChange(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="">Select Weather</option>
            {weatherConditions.map((w) => (
              <option key={w} value={w}>{w}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="space-y-4">
        <label className="flex items-center space-x-2 text-sm font-semibold text-gray-700">
          <Users className="w-4 h-4" />
          <span>Toss Winner</span>
        </label>
        <div className="flex space-x-3">
          <button
            onClick={() => onTossWinnerChange('team1')}
            className={`flex-1 p-3 rounded-lg border-2 transition-all ${
              tossWinner === 'team1'
                ? 'border-blue-500 bg-blue-50 text-blue-700'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            {team1Name || 'Team 1'}
          </button>
          <button
            onClick={() => onTossWinnerChange('team2')}
            className={`flex-1 p-3 rounded-lg border-2 transition-all ${
              tossWinner === 'team2'
                ? 'border-blue-500 bg-blue-50 text-blue-700'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            {team2Name || 'Team 2'}
          </button>
        </div>

        {tossWinner && (
          <div className="space-y-2">
            <label className="text-sm font-semibold text-gray-700">Toss Decision</label>
            <div className="flex space-x-3">
              <button
                onClick={() => onTossDecisionChange('bat')}
                className={`flex-1 p-3 rounded-lg border-2 transition-all ${
                  tossDecision === 'bat'
                    ? 'border-green-500 bg-green-50 text-green-700'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                Bat First
              </button>
              <button
                onClick={() => onTossDecisionChange('bowl')}
                className={`flex-1 p-3 rounded-lg border-2 transition-all ${
                  tossDecision === 'bowl'
                    ? 'border-green-500 bg-green-50 text-green-700'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                Bowl First
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}