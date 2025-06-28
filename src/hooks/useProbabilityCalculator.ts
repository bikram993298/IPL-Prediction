import { useState, useEffect } from 'react';
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

  useEffect(() => {
    if (!matchState.team1 || !matchState.team2) {
      setProbability({ team1: 50, team2: 50, confidence: 'low' });
      return;
    }

    const calculateProbability = () => {
      let team1Prob = 50;
      let confidence: 'low' | 'medium' | 'high' = 'low';
      
      const totalBalls = matchState.overs * 6 + matchState.balls;
      const remainingBalls = 120 - totalBalls;
      
      // Base probability adjustments
      if (matchState.isFirstInnings) {
        // First innings - batting team probability
        const runRate = totalBalls > 0 ? (matchState.currentScore / totalBalls) * 6 : 0;
        
        if (runRate > 8.5) team1Prob += 15;
        else if (runRate > 7) team1Prob += 10;
        else if (runRate > 6) team1Prob += 5;
        else if (runRate < 4) team1Prob -= 15;
        else if (runRate < 5) team1Prob -= 10;
        
        // Wickets factor
        if (matchState.wickets >= 7) team1Prob -= 20;
        else if (matchState.wickets >= 5) team1Prob -= 10;
        else if (matchState.wickets <= 2) team1Prob += 5;
        
        // Overs remaining factor
        if (remainingBalls > 60 && matchState.wickets <= 3) team1Prob += 10;
        
      } else {
        // Second innings - chasing team probability
        if (matchState.target > 0) {
          const required = matchState.target - matchState.currentScore;
          const requiredRate = remainingBalls > 0 ? (required / remainingBalls) * 6 : 0;
          
          if (required <= 0) {
            team1Prob = 95; // Almost certain win
          } else if (requiredRate > 12) {
            team1Prob = 15; // Very difficult
          } else if (requiredRate > 10) {
            team1Prob = 25;
          } else if (requiredRate > 8) {
            team1Prob = 35;
          } else if (requiredRate > 6) {
            team1Prob = 55;
          } else if (requiredRate > 4) {
            team1Prob = 70;
          } else {
            team1Prob = 80; // Easy chase
          }
          
          // Wickets in hand adjustment
          const wicketsInHand = 10 - matchState.wickets;
          if (wicketsInHand >= 7) team1Prob += 10;
          else if (wicketsInHand >= 5) team1Prob += 5;
          else if (wicketsInHand <= 2) team1Prob -= 20;
          else if (wicketsInHand <= 3) team1Prob -= 10;
          
          // Balls remaining factor
          if (remainingBalls < 12 && required > 15) team1Prob -= 15;
          else if (remainingBalls < 24 && required > 30) team1Prob -= 10;
          
          confidence = remainingBalls < 30 ? 'high' : remainingBalls < 60 ? 'medium' : 'low';
        }
      }
      
      // Toss advantage
      if (matchState.tossWinner === 'team1' && matchState.tossDecision === 'bat') {
        team1Prob += 3;
      } else if (matchState.tossWinner === 'team2' && matchState.tossDecision === 'bowl') {
        team1Prob -= 3;
      }
      
      // Venue advantage (simplified)
      const homeAdvantage: Record<string, string> = {
        'Wankhede Stadium, Mumbai': 'mi',
        'Eden Gardens, Kolkata': 'kkr',
        'M. Chinnaswamy Stadium, Bangalore': 'rcb',
        'MA Chidambaram Stadium, Chennai': 'csk',
        'Arun Jaitley Stadium, Delhi': 'dc',
        'Sawai Mansingh Stadium, Jaipur': 'rr',
        'PCA Stadium, Mohali': 'pbks',
        'Rajiv Gandhi International Stadium, Hyderabad': 'srh',
      };
      
      if (homeAdvantage[matchState.venue] === matchState.team1.id) {
        team1Prob += 5;
      } else if (homeAdvantage[matchState.venue] === matchState.team2.id) {
        team1Prob -= 5;
      }
      
      // Weather conditions
      if (matchState.weather === 'Light Rain' || matchState.weather === 'Overcast') {
        team1Prob -= 2; // Slightly favors bowling team
      } else if (matchState.weather === 'Dew Expected') {
        if (!matchState.isFirstInnings) team1Prob += 5; // Helps chasing team
      }
      
      // Ensure probability is within bounds
      team1Prob = Math.max(5, Math.min(95, team1Prob));
      
      // Calculate factors
      const newFactors: MatchFactors = {
        momentum: Math.max(1, Math.min(10, 
          matchState.isFirstInnings 
            ? (totalBalls > 0 ? Math.round((matchState.currentScore / totalBalls) * 6 / 2) : 5)
            : (matchState.target > 0 ? Math.round(10 - Math.abs(((matchState.target - matchState.currentScore) / remainingBalls * 6) - 6)) : 5)
        )),
        pressure: Math.max(1, Math.min(10, 
          matchState.isFirstInnings 
            ? Math.round(matchState.wickets + 2)
            : Math.round((10 - matchState.wickets) / 2 + remainingBalls / 20)
        )),
        form: Math.floor(Math.random() * 3) + 6, // Simplified random form factor
        conditions: Math.max(1, Math.min(10, 
          ['Clear', 'Partly Cloudy'].includes(matchState.weather) ? 8 : 
          matchState.weather === 'Overcast' ? 6 : 4
        ))
      };
      
      setFactors(newFactors);
      setProbability({
        team1: team1Prob,
        team2: 100 - team1Prob,
        confidence
      });
    };

    calculateProbability();
  }, [matchState]);

  return { probability, factors };
}