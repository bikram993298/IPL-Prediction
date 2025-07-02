import { useState, useEffect } from 'react';
import { mlApi } from '../services/mlApi';
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

interface Probability {
  team1: number;
  team2: number;
  confidence: 'low' | 'medium' | 'high';
}

interface MatchFactors {
  momentum: number;
  pressure: number;
  form: number;
  conditions: number;
}

export function useProbabilityCalculator(matchState: MatchState) {
  const [probability, setProbability] = useState<Probability>({ team1: 50, team2: 50, confidence: 'low' });
  const [factors, setFactors] = useState<MatchFactors>({ momentum: 5, pressure: 5, form: 5, conditions: 5 });
  const [isLoading, setIsLoading] = useState(false);
  const [usingMLBackend, setUsingMLBackend] = useState(false);

  useEffect(() => {
    if (!matchState.team1 || !matchState.team2) {
      setProbability({ team1: 50, team2: 50, confidence: 'low' });
      setFactors({ momentum: 5, pressure: 5, form: 5, conditions: 5 });
      return;
    }

    const calculateProbability = async () => {
      setIsLoading(true);
      
      try {
        // Prepare ML request
        const mlRequest = {
          team1: matchState.team1!.name,
          team2: matchState.team2!.name,
          current_score: matchState.currentScore,
          wickets: matchState.wickets,
          overs: matchState.overs,
          balls: matchState.balls,
          target: matchState.isFirstInnings ? undefined : matchState.target,
          venue: matchState.venue || 'Wankhede Stadium, Mumbai',
          weather: matchState.weather || 'Clear',
          toss_winner: matchState.tossWinner === 'team1' ? matchState.team1!.name : 
                      matchState.tossWinner === 'team2' ? matchState.team2!.name : '',
          toss_decision: matchState.tossDecision || '',
          is_first_innings: matchState.isFirstInnings,
        };

        // Get ML prediction
        const mlResponse = await mlApi.predict(mlRequest);
        
        // Check if we're using ML backend or fallback
        const isUsingML = await mlApi.checkConnection();
        setUsingMLBackend(isUsingML);
        
        // Update probability
        setProbability({
          team1: mlResponse.team1_probability,
          team2: mlResponse.team2_probability,
          confidence: mlResponse.confidence
        });
        
        // Update factors
        setFactors(mlResponse.factors);
        
      } catch (error) {
        console.error('Prediction error:', error);
        // Fallback to basic calculation
        setUsingMLBackend(false);
        setProbability({ team1: 50, team2: 50, confidence: 'low' });
        setFactors({ momentum: 5, pressure: 5, form: 5, conditions: 5 });
      } finally {
        setIsLoading(false);
      }
    };

    calculateProbability();
  }, [matchState]);

  return { 
    probability, 
    factors, 
    isLoading, 
    usingMLBackend 
  };
}