import React, { useState } from 'react';
import { Trophy, BarChart3, Calendar, Brain } from 'lucide-react';
import TeamSelector, { Team } from './components/TeamSelector';
import MatchDetails from './components/MatchDetails';
import ScoreInput from './components/ScoreInput';
import ProbabilityDisplay from './components/ProbabilityDisplay';
import MatchFactors from './components/MatchFactors';
import LiveMatchIntegration from './components/LiveMatchIntegration';
import MLBackendStatus from './components/MLBackendStatus';
import { useProbabilityCalculator } from './hooks/useProbabilityCalculator';

function App() {
  const [team1, setTeam1] = useState<Team | null>(null);
  const [team2, setTeam2] = useState<Team | null>(null);
  const [currentScore, setCurrentScore] = useState(0);
  const [wickets, setWickets] = useState(0);
  const [overs, setOvers] = useState(0);
  const [balls, setBalls] = useState(0);
  const [target, setTarget] = useState(0);
  const [venue, setVenue] = useState('');
  const [tossWinner, setTossWinner] = useState<'team1' | 'team2' | null>(null);
  const [tossDecision, setTossDecision] = useState<'bat' | 'bowl' | null>(null);
  const [weather, setWeather] = useState('');
  const [isFirstInnings, setIsFirstInnings] = useState(true);
  const [showLiveIntegration, setShowLiveIntegration] = useState(false);

  const matchState = {
    team1,
    team2,
    currentScore,
    wickets,
    overs,
    balls,
    target,
    venue,
    tossWinner,
    tossDecision,
    weather,
    isFirstInnings,
  };

  const { probability, factors, isLoading, usingMLBackend } = useProbabilityCalculator(matchState);

  const handleLiveMatchUpdate = (data: any) => {
    setTeam1(data.team1);
    setTeam2(data.team2);
    setCurrentScore(data.currentScore);
    setWickets(data.wickets);
    setOvers(data.overs);
    setBalls(data.balls);
    setTarget(data.target);
    setVenue(data.venue);
    setTossWinner(data.tossWinner);
    setTossDecision(data.tossDecision);
    setIsFirstInnings(data.isFirstInnings);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                <Trophy className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">IPL Win Predictor</h1>
                <p className="text-sm text-gray-600">
                  Full-stack ML-powered cricket analytics with local Python backend
                </p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <Calendar className="w-4 h-4" />
                <span>IPL 2024 Season</span>
              </div>
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <Brain className="w-4 h-4" />
                <span className={usingMLBackend ? 'text-green-600' : 'text-orange-600'}>
                  {usingMLBackend ? 'ML Backend Active' : 'Fallback Mode'}
                </span>
              </div>
              <button
                onClick={() => setShowLiveIntegration(!showLiveIntegration)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  showLiveIntegration 
                    ? 'bg-green-100 text-green-700 border border-green-300' 
                    : 'bg-blue-100 text-blue-700 border border-blue-300'
                }`}
              >
                {showLiveIntegration ? 'Hide Live Data' : 'Live Data'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* ML Backend Status */}
          <MLBackendStatus />

          {/* Live Match Integration */}
          {showLiveIntegration && (
            <LiveMatchIntegration onMatchDataUpdate={handleLiveMatchUpdate} />
          )}

          {/* Team Selection */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center space-x-2 mb-6">
              <Trophy className="w-5 h-5 text-blue-600" />
              <h2 className="text-xl font-bold text-gray-800">Select Teams</h2>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <TeamSelector
                selectedTeam={team1}
                onTeamSelect={setTeam1}
                label="Team 1 (Batting)"
              />
              <TeamSelector
                selectedTeam={team2}
                onTeamSelect={setTeam2}
                label="Team 2 (Bowling)"
              />
            </div>
          </div>

          {/* Match Details */}
          <MatchDetails
            venue={venue}
            onVenueChange={setVenue}
            tossWinner={tossWinner}
            onTossWinnerChange={setTossWinner}
            tossDecision={tossDecision}
            onTossDecisionChange={setTossDecision}
            weather={weather}
            onWeatherChange={setWeather}
            team1Name={team1?.shortName || ''}
            team2Name={team2?.shortName || ''}
          />

          {/* Innings Toggle */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-bold text-gray-800">Match Phase</h3>
              <div className="flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setIsFirstInnings(true)}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                    isFirstInnings
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-gray-600 hover:text-gray-800'
                  }`}
                >
                  First Innings
                </button>
                <button
                  onClick={() => setIsFirstInnings(false)}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                    !isFirstInnings
                      ? 'bg-blue-600 text-white shadow-sm'
                      : 'text-gray-600 hover:text-gray-800'
                  }`}
                >
                  Second Innings
                </button>
              </div>
            </div>
          </div>

          {/* Score Input */}
          <ScoreInput
            currentScore={currentScore}
            onCurrentScoreChange={setCurrentScore}
            wickets={wickets}
            onWicketsChange={setWickets}
            overs={overs}
            onOversChange={setOvers}
            balls={balls}
            onBallsChange={setBalls}
            target={target}
            onTargetChange={setTarget}
            isFirstInnings={isFirstInnings}
          />

          {/* Results Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2">
              <ProbabilityDisplay
                team1={team1}
                team2={team2}
                probability={probability}
                isLoading={isLoading}
                usingMLBackend={usingMLBackend}
              />
            </div>
            <div>
              <MatchFactors factors={factors} />
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-50 border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-sm text-gray-600">
            <p>IPL Win Probability Predictor • Full-Stack ML Analytics</p>
            <p className="mt-1">
              Frontend: React + TypeScript • Backend: Python + FastAPI • 
              {usingMLBackend ? ' 🚀 Local ML Active' : ' ⚡ Fallback Mode'}
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;