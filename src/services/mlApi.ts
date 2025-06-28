// ML Backend API Integration
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

  async checkConnection(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      this.isConnected = response.ok;
      return this.isConnected;
    } catch (error) {
      console.warn('ML Backend not available:', error);
      this.isConnected = false;
      return false;
    }
  }

  async predict(request: MLPredictionRequest): Promise<MLPredictionResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`ML API error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('ML prediction error:', error);
      // Fallback to basic calculation
      return this.fallbackPrediction(request);
    }
  }

  private fallbackPrediction(request: MLPredictionRequest): MLPredictionResponse {
    // Simple fallback calculation when ML backend is unavailable
    let team1Prob = 50;
    
    if (!request.is_first_innings && request.target) {
      const required = request.target - request.current_score;
      const ballsRemaining = 120 - (request.overs * 6 + request.balls);
      const requiredRate = ballsRemaining > 0 ? (required / ballsRemaining) * 6 : 0;
      
      if (required <= 0) {
        team1Prob = 95;
      } else if (requiredRate > 12) {
        team1Prob = 15;
      } else if (requiredRate > 10) {
        team1Prob = 25;
      } else if (requiredRate > 8) {
        team1Prob = 40;
      } else if (requiredRate > 6) {
        team1Prob = 60;
      } else {
        team1Prob = 80;
      }
      
      // Adjust for wickets
      const wicketsInHand = 10 - request.wickets;
      if (wicketsInHand <= 2) team1Prob -= 20;
      else if (wicketsInHand <= 4) team1Prob -= 10;
      else if (wicketsInHand >= 8) team1Prob += 10;
    }
    
    team1Prob = Math.max(5, Math.min(95, team1Prob));
    
    return {
      team1_probability: team1Prob,
      team2_probability: 100 - team1Prob,
      confidence: 'medium',
      factors: {
        momentum: 5,
        pressure: 5,
        form: 5,
        conditions: 5,
      },
      model_predictions: {
        fallback: team1Prob / 100,
      },
      feature_importance: {},
      prediction_timestamp: new Date().toISOString(),
    };
  }

  async getModelPerformance(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/model-performance`);
      return await response.json();
    } catch (error) {
      console.error('Failed to get model performance:', error);
      return null;
    }
  }

  async getAvailableModels(): Promise<MLModelInfo[]> {
    try {
      const response = await fetch(`${this.baseUrl}/models`);
      const data = await response.json();
      return data.models || [];
    } catch (error) {
      console.error('Failed to get available models:', error);
      return [];
    }
  }

  async trainNewModel(modelType: string, hyperparameters: Record<string, any>): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/train-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_type: modelType,
          hyperparameters,
          training_data_size: 10000,
        }),
      });

      return await response.json();
    } catch (error) {
      console.error('Failed to train model:', error);
      throw error;
    }
  }

  async getFeatureImportance(): Promise<Record<string, number>> {
    try {
      const response = await fetch(`${this.baseUrl}/feature-importance`);
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
}

export const mlApi = new MLApiService();