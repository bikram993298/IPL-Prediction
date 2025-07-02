// ML Backend API Integration - Local Backend
export interface MLPredictionRequest {
  team1: string;
  team2: string;
  current_score: number;
  wickets: number;
  overs: number;
  balls: number;
  target?: number;
  venue: string;
  weather: string;
  toss_winner: string;
  toss_decision: string;
  is_first_innings: boolean;
}

export interface MLPredictionResponse {
  team1_probability: number;
  team2_probability: number;
  confidence: 'low' | 'medium' | 'high';
  factors: {
    momentum: number;
    pressure: number;
    form: number;
    conditions: number;
  };
  model_predictions: Record<string, number>;
  feature_importance: Record<string, number>;
  prediction_timestamp: string;
}

export interface MLModelInfo {
  name: string;
  type: string;
  created_at: string;
  training_size: number;
  is_loaded: boolean;
}

class MLApiService {
  private baseUrl = 'http://localhost:8000';
  private isConnected = false;
  private connectionChecked = false;

  async checkConnection(): Promise<boolean> {
    if (this.connectionChecked) {
      return this.isConnected;
    }

    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      this.isConnected = response.ok;
      this.connectionChecked = true;
      
      if (this.isConnected) {
        console.log('✅ ML Backend connected successfully');
      } else {
        console.warn('⚠️ ML Backend responded but not healthy');
      }
      
      return this.isConnected;
    } catch (error) {
      console.warn('❌ ML Backend not available:', error);
      this.isConnected = false;
      this.connectionChecked = true;
      return false;
    }
  }

  async predict(request: MLPredictionRequest): Promise<MLPredictionResponse> {
    const isConnected = await this.checkConnection();
    
    if (!isConnected) {
      console.log('🔄 Using fallback prediction (ML backend offline)');
      return this.fallbackPrediction(request);
    }

    try {
      console.log('🧠 Making ML prediction request...');
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`ML API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('✅ ML prediction successful');
      return result;
    } catch (error) {
      console.error('❌ ML prediction error:', error);
      console.log('🔄 Falling back to basic calculation');
      return this.fallbackPrediction(request);
    }
  }

  private fallbackPrediction(request: MLPredictionRequest): MLPredictionResponse {
    console.log('🔧 Calculating fallback prediction...');
    
    // Advanced fallback calculation
    let team1Prob = 50;
    
    const totalBalls = request.overs * 6 + request.balls;
    const ballsRemaining = 120 - totalBalls;
    
    if (!request.is_first_innings && request.target) {
      // Second innings - chasing
      const required = request.target - request.current_score;
      const requiredRate = ballsRemaining > 0 ? (required / ballsRemaining) * 6 : 0;
      
      if (required <= 0) {
        team1Prob = 95; // Already won
      } else if (requiredRate > 15) {
        team1Prob = 5; // Nearly impossible
      } else if (requiredRate > 12) {
        team1Prob = 15; // Very difficult
      } else if (requiredRate > 10) {
        team1Prob = 25; // Difficult
      } else if (requiredRate > 8) {
        team1Prob = 40; // Challenging
      } else if (requiredRate > 6) {
        team1Prob = 60; // Achievable
      } else if (requiredRate > 4) {
        team1Prob = 75; // Comfortable
      } else {
        team1Prob = 85; // Easy
      }
      
      // Adjust for wickets in hand
      const wicketsInHand = 10 - request.wickets;
      if (wicketsInHand <= 1) team1Prob -= 25;
      else if (wicketsInHand <= 3) team1Prob -= 15;
      else if (wicketsInHand <= 5) team1Prob -= 5;
      else if (wicketsInHand >= 8) team1Prob += 10;
      
      // Adjust for balls remaining
      if (ballsRemaining < 6 && required > 10) team1Prob -= 20;
      else if (ballsRemaining < 12 && required > 20) team1Prob -= 15;
      else if (ballsRemaining < 24 && required > 40) team1Prob -= 10;
      
    } else {
      // First innings - batting
      const currentRate = totalBalls > 0 ? (request.current_score / totalBalls) * 6 : 0;
      
      // Adjust based on run rate
      if (currentRate > 10) team1Prob += 20;
      else if (currentRate > 8) team1Prob += 15;
      else if (currentRate > 7) team1Prob += 10;
      else if (currentRate > 6) team1Prob += 5;
      else if (currentRate < 4) team1Prob -= 20;
      else if (currentRate < 5) team1Prob -= 10;
      
      // Adjust for wickets lost
      if (request.wickets >= 8) team1Prob -= 25;
      else if (request.wickets >= 6) team1Prob -= 15;
      else if (request.wickets >= 4) team1Prob -= 5;
      else if (request.wickets <= 2) team1Prob += 10;
      
      // Adjust for overs remaining
      if (ballsRemaining > 60 && request.wickets <= 3) team1Prob += 15;
      else if (ballsRemaining < 30 && request.wickets >= 6) team1Prob -= 10;
    }
    
    // Venue advantage
    const homeAdvantage: Record<string, string> = {
      'Wankhede Stadium, Mumbai': 'Mumbai Indians',
      'Eden Gardens, Kolkata': 'Kolkata Knight Riders',
      'M. Chinnaswamy Stadium, Bangalore': 'Royal Challengers Bangalore',
      'MA Chidambaram Stadium, Chennai': 'Chennai Super Kings',
      'Arun Jaitley Stadium, Delhi': 'Delhi Capitals',
      'Sawai Mansingh Stadium, Jaipur': 'Rajasthan Royals',
      'PCA Stadium, Mohali': 'Punjab Kings',
      'Rajiv Gandhi International Stadium, Hyderabad': 'Sunrisers Hyderabad',
    };
    
    if (homeAdvantage[request.venue] === request.team1) {
      team1Prob += 5;
    } else if (homeAdvantage[request.venue] === request.team2) {
      team1Prob -= 5;
    }
    
    // Weather conditions
    if (request.weather === 'Light Rain' || request.weather === 'Overcast') {
      team1Prob -= 3; // Favors bowling
    } else if (request.weather === 'Dew Expected' && !request.is_first_innings) {
      team1Prob += 5; // Helps chasing team
    }
    
    // Toss advantage
    if (request.toss_winner === request.team1 && request.toss_decision === 'bat') {
      team1Prob += 3;
    } else if (request.toss_winner === request.team2 && request.toss_decision === 'bowl') {
      team1Prob -= 3;
    }
    
    // Ensure probability is within bounds
    team1Prob = Math.max(5, Math.min(95, team1Prob));
    
    // Calculate factors
    const momentum = Math.max(1, Math.min(10, 
      request.is_first_innings 
        ? (totalBalls > 0 ? Math.round((request.current_score / totalBalls) * 6 / 1.5) : 5)
        : (request.target ? Math.round(10 - Math.abs(((request.target - request.current_score) / ballsRemaining * 6) - 6)) : 5)
    ));
    
    const pressure = Math.max(1, Math.min(10, 
      request.is_first_innings 
        ? Math.round(request.wickets + 2)
        : Math.round((10 - request.wickets) / 2 + ballsRemaining / 20)
    ));
    
    const form = Math.floor(Math.random() * 3) + 6; // Random form factor
    
    const conditions = Math.max(1, Math.min(10, 
      ['Clear', 'Partly Cloudy'].includes(request.weather) ? 8 : 
      request.weather === 'Overcast' ? 6 : 4
    ));
    
    return {
      team1_probability: team1Prob,
      team2_probability: 100 - team1Prob,
      confidence: ballsRemaining < 30 ? 'high' : ballsRemaining < 60 ? 'medium' : 'low',
      factors: {
        momentum,
        pressure,
        form,
        conditions,
      },
      model_predictions: {
        fallback_algorithm: team1Prob / 100,
      },
      feature_importance: {
        'required_run_rate': 0.25,
        'wickets_in_hand': 0.20,
        'balls_remaining': 0.15,
        'current_run_rate': 0.15,
        'venue_advantage': 0.10,
        'weather_conditions': 0.08,
        'toss_advantage': 0.07,
      },
      prediction_timestamp: new Date().toISOString(),
    };
  }

  async getModelPerformance(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/model-performance`);
      if (!response.ok) throw new Error('Performance data unavailable');
      return await response.json();
    } catch (error) {
      console.error('Failed to get model performance:', error);
      return {
        status: 'offline',
        total_predictions: 0,
        average_processing_time_ms: 0,
        confidence_distribution: { high: 0, medium: 0, low: 0 }
      };
    }
  }

  async getAvailableModels(): Promise<MLModelInfo[]> {
    try {
      const response = await fetch(`${this.baseUrl}/models`);
      if (!response.ok) throw new Error('Models data unavailable');
      const data = await response.json();
      return data.models || [];
    } catch (error) {
      console.error('Failed to get available models:', error);
      return [];
    }
  }

  async getFeatureImportance(): Promise<Record<string, number>> {
    try {
      const response = await fetch(`${this.baseUrl}/feature-importance`);
      if (!response.ok) throw new Error('Feature importance unavailable');
      const data = await response.json();
      return data.feature_importance || {};
    } catch (error) {
      console.error('Failed to get feature importance:', error);
      return {};
    }
  }

  isMLBackendAvailable(): boolean {
    return this.isConnected;
  }

  // Reset connection status (useful for retry logic)
  resetConnection(): void {
    this.connectionChecked = false;
    this.isConnected = false;
  }
}

export const mlApi = new MLApiService();