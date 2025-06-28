import { useState, useEffect } from 'react';
import { mlApi, MLPredictionRequest, MLPredictionResponse } from '../services/mlApi';
import type { Team } from '../components/TeamSelector';

interface MatchState {
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
  weather: string;
  isFirstInnings: boolean;
}

export function useMLPrediction(matchState: MatchState) {
  const [prediction, setPrediction] = useState<MLPredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mlBackendAvailable, setMlBackendAvailable] = useState(false);

  // Check ML backend availability on mount
  useEffect(() => {
    const checkBackend = async () => {
      const available = await mlApi.checkConnection();
      setMlBackendAvailable(available);
    };
    
    checkBackend();
  }, []);

  useEffect(() => {
    if (!matchState.team1 || !matchState.team2) {
      setPrediction(null);
      return;
    }

    const makePrediction = async () => {
      setLoading(true);
      setError(null);

      try {
        const request: MLPredictionRequest = {
          team1: matchState.team1!.name,
          team2: matchState.team2!.name,
          current_score: matchState.currentScore,
          wickets: matchState.wickets,
          overs: matchState.overs,
          balls: matchState.balls,
          target: matchState.isFirstInnings ? undefined : matchState.target,
          venue: matchState.venue,
          weather: matchState.weather,
          toss_winner: matchState.tossWinner === 'team1' ? matchState.team1!.name : 
                      matchState.tossWinner === 'team2' ? matchState.team2!.name : '',
          toss_decision: matchState.tossDecision || '',
          is_first_innings: matchState.isFirstInnings,
        };

        const result = await mlApi.predict(request);
        setPrediction(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Prediction failed');
      } finally {
        setLoading(false);
      }
    };

    makePrediction();
  }, [matchState]);

  return {
    prediction,
    loading,
    error,
    mlBackendAvailable,
  };
}