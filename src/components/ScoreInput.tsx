import React from 'react';
import { Target, TrendingUp, Clock, Users } from 'lucide-react';

interface ScoreInputProps {
  currentScore: number;
  onCurrentScoreChange: (score: number) => void;
  wickets: number;
  onWicketsChange: (wickets: number) => void;
  overs: number;
  onOversChange: (overs: number) => void;
  balls: number;
  onBallsChange: (balls: number) => void;
  target: number;
  onTargetChange: (target: number) => void;
  isFirstInnings: boolean;
}

export default function ScoreInput({
  currentScore,
  onCurrentScoreChange,
  wickets,
  onWicketsChange,
  overs,
  onOversChange,
  balls,
  onBallsChange,
  target,
  onTargetChange,
  isFirstInnings,
}: ScoreInputProps) {
  const totalBalls = overs * 6 + balls;
  const remainingBalls = 120 - totalBalls;
  const requiredRunRate = target > 0 && remainingBalls > 0 ? ((target - currentScore) / remainingBalls * 6).toFixed(2) : '0.00';
  const currentRunRate = totalBalls > 0 ? (currentScore / totalBalls * 6).toFixed(2) : '0.00';

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 space-y-6">
      <div className="flex items-center space-x-2 mb-4">
        <TrendingUp className="w-5 h-5 text-green-600" />
        <h3 className="text-lg font-bold text-gray-800">Current Match Situation</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="space-y-2">
          <label className="text-sm font-semibold text-gray-700">Current Score</label>
          <input
            type="number"
            value={currentScore}
            onChange={(e) => onCurrentScoreChange(Number(e.target.value))}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            min="0"
            max="400"
          />
        </div>

        <div className="space-y-2">
          <label className="flex items-center space-x-1 text-sm font-semibold text-gray-700">
            <Users className="w-4 h-4" />
            <span>Wickets</span>
          </label>
          <input
            type="number"
            value={wickets}
            onChange={(e) => onWicketsChange(Number(e.target.value))}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            min="0"
            max="10"
          />
        </div>

        <div className="space-y-2">
          <label className="flex items-center space-x-1 text-sm font-semibold text-gray-700">
            <Clock className="w-4 h-4" />
            <span>Overs</span>
          </label>
          <input
            type="number"
            value={overs}
            onChange={(e) => onOversChange(Number(e.target.value))}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            min="0"
            max="20"
          />
        </div>

        <div className="space-y-2">
          <label className="text-sm font-semibold text-gray-700">Balls in Over</label>
          <input
            type="number"
            value={balls}
            onChange={(e) => onBallsChange(Number(e.target.value))}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            min="0"
            max="5"
          />
        </div>
      </div>

      {!isFirstInnings && (
        <div className="space-y-2">
          <label className="flex items-center space-x-1 text-sm font-semibold text-gray-700">
            <Target className="w-4 h-4" />
            <span>Target</span>
          </label>
          <input
            type="number"
            value={target}
            onChange={(e) => onTargetChange(Number(e.target.value))}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            min="0"
            max="400"
          />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t">
        <div className="text-center p-3 bg-blue-50 rounded-lg">
          <div className="text-sm text-gray-600">Current Run Rate</div>
          <div className="text-xl font-bold text-blue-600">{currentRunRate}</div>
        </div>
        
        {!isFirstInnings && target > 0 && (
          <>
            <div className="text-center p-3 bg-orange-50 rounded-lg">
              <div className="text-sm text-gray-600">Required Run Rate</div>
              <div className="text-xl font-bold text-orange-600">{requiredRunRate}</div>
            </div>
            
            <div className="text-center p-3 bg-purple-50 rounded-lg">
              <div className="text-sm text-gray-600">Runs Required</div>
              <div className="text-xl font-bold text-purple-600">{Math.max(0, target - currentScore)}</div>
            </div>
          </>
        )}
        
        {isFirstInnings && (
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-sm text-gray-600">Balls Remaining</div>
            <div className="text-xl font-bold text-green-600">{remainingBalls}</div>
          </div>
        )}
      </div>
    </div>
  );
}