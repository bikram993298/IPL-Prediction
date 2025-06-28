// Cricket API service for real-time data integration
export interface LiveMatch {
  matchId: string;
  team1: string;
  team2: string;
  currentScore: number;
  wickets: number;
  overs: number;
  balls: number;
  target?: number;
  status: 'live' | 'completed' | 'upcoming';
  venue: string;
  tossWinner?: string;
  tossDecision?: 'bat' | 'bowl';
  battingTeam: string;
  bowlingTeam: string;
  lastUpdated: string;
}

export interface BallByBallData {
  over: number;
  ball: number;
  runs: number;
  isWicket: boolean;
  batsman: string;
  bowler: string;
  commentary: string;
}

class CricketApiService {
  private apiKey = '77801300-cbed-4c96-869a-81e31ebc1484';
  private baseUrl = 'https://api.cricapi.com/v1';

  // Method 1: CricAPI with your API key
  async getCricApiMatches(): Promise<LiveMatch[]> {
    try {
      const response = await fetch(`${this.baseUrl}/currentMatches?apikey=${this.apiKey}&offset=0`);
      const data = await response.json();
      
      if (!data.success) {
        console.warn('CricAPI response not successful:', data.reason);
        return [];
      }
      
      return data.data?.filter((match: any) => 
        match.matchType === 'T20' && 
        (match.series?.includes('Indian Premier League') || 
         match.teams?.some((team: string) => this.isIPLTeam(team)))
      ).map((match: any) => ({
        matchId: match.id,
        team1: this.normalizeTeamName(match.teams[0]),
        team2: this.normalizeTeamName(match.teams[1]),
        currentScore: this.parseScore(match.score?.[0]?.r) || 0,
        wickets: this.parseScore(match.score?.[0]?.w) || 0,
        overs: Math.floor(this.parseScore(match.score?.[0]?.o) || 0),
        balls: Math.round(((this.parseScore(match.score?.[0]?.o) || 0) % 1) * 6),
        target: this.parseScore(match.score?.[1]?.r) || undefined,
        status: this.normalizeStatus(match.status),
        venue: match.venue || 'Unknown Venue',
        tossWinner: match.tossWinner ? this.normalizeTeamName(match.tossWinner) : undefined,
        tossDecision: match.tossChoice?.toLowerCase() === 'bat' ? 'bat' : 
                     match.tossChoice?.toLowerCase() === 'bowl' ? 'bowl' : undefined,
        battingTeam: this.normalizeTeamName(match.teams[0]),
        bowlingTeam: this.normalizeTeamName(match.teams[1]),
        lastUpdated: new Date().toISOString()
      })) || [];
    } catch (error) {
      console.error('Error fetching from CricAPI:', error);
      return [];
    }
  }

  // Method 2: Get specific match details
  async getMatchDetails(matchId: string): Promise<LiveMatch | null> {
    try {
      const response = await fetch(`${this.baseUrl}/match_info?apikey=${this.apiKey}&id=${matchId}`);
      const data = await response.json();
      
      if (!data.success) {
        console.warn('Match details not available:', data.reason);
        return null;
      }
      
      const match = data.data;
      return {
        matchId: match.id,
        team1: this.normalizeTeamName(match.teams[0]),
        team2: this.normalizeTeamName(match.teams[1]),
        currentScore: this.parseScore(match.score?.[0]?.r) || 0,
        wickets: this.parseScore(match.score?.[0]?.w) || 0,
        overs: Math.floor(this.parseScore(match.score?.[0]?.o) || 0),
        balls: Math.round(((this.parseScore(match.score?.[0]?.o) || 0) % 1) * 6),
        target: this.parseScore(match.score?.[1]?.r) || undefined,
        status: this.normalizeStatus(match.status),
        venue: match.venue || 'Unknown Venue',
        tossWinner: match.tossWinner ? this.normalizeTeamName(match.tossWinner) : undefined,
        tossDecision: match.tossChoice?.toLowerCase() === 'bat' ? 'bat' : 
                     match.tossChoice?.toLowerCase() === 'bowl' ? 'bowl' : undefined,
        battingTeam: this.normalizeTeamName(match.teams[0]),
        bowlingTeam: this.normalizeTeamName(match.teams[1]),
        lastUpdated: new Date().toISOString()
      };
    } catch (error) {
      console.error('Error fetching match details:', error);
      return null;
    }
  }

  // Helper method to check if team is IPL team
  private isIPLTeam(teamName: string): boolean {
    const iplTeams = [
      'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
      'Kolkata Knight Riders', 'Delhi Capitals', 'Rajasthan Royals',
      'Punjab Kings', 'Sunrisers Hyderabad', 'CSK', 'MI', 'RCB', 'KKR',
      'DC', 'RR', 'PBKS', 'SRH', 'Chennai', 'Mumbai', 'Bangalore',
      'Kolkata', 'Delhi', 'Rajasthan', 'Punjab', 'Hyderabad'
    ];
    
    return iplTeams.some(team => 
      teamName.toLowerCase().includes(team.toLowerCase()) ||
      team.toLowerCase().includes(teamName.toLowerCase())
    );
  }

  // Helper method to normalize team names
  private normalizeTeamName(teamName: string): string {
    const teamMappings: Record<string, string> = {
      'CSK': 'Chennai Super Kings',
      'MI': 'Mumbai Indians',
      'RCB': 'Royal Challengers Bangalore',
      'KKR': 'Kolkata Knight Riders',
      'DC': 'Delhi Capitals',
      'RR': 'Rajasthan Royals',
      'PBKS': 'Punjab Kings',
      'SRH': 'Sunrisers Hyderabad',
      'Chennai': 'Chennai Super Kings',
      'Mumbai': 'Mumbai Indians',
      'Bangalore': 'Royal Challengers Bangalore',
      'Kolkata': 'Kolkata Knight Riders',
      'Delhi': 'Delhi Capitals',
      'Rajasthan': 'Rajasthan Royals',
      'Punjab': 'Punjab Kings',
      'Hyderabad': 'Sunrisers Hyderabad'
    };

    // Check for exact matches first
    if (teamMappings[teamName]) {
      return teamMappings[teamName];
    }

    // Check for partial matches
    for (const [key, value] of Object.entries(teamMappings)) {
      if (teamName.toLowerCase().includes(key.toLowerCase()) ||
          key.toLowerCase().includes(teamName.toLowerCase())) {
        return value;
      }
    }

    return teamName; // Return original if no mapping found
  }

  // Helper method to normalize match status
  private normalizeStatus(status: string): 'live' | 'completed' | 'upcoming' {
    const statusLower = status.toLowerCase();
    
    if (statusLower.includes('live') || statusLower.includes('progress')) {
      return 'live';
    } else if (statusLower.includes('finished') || statusLower.includes('completed') || statusLower.includes('won')) {
      return 'completed';
    } else {
      return 'upcoming';
    }
  }

  // Helper method to safely parse scores
  private parseScore(value: any): number {
    if (typeof value === 'number') return value;
    if (typeof value === 'string') {
      const parsed = parseInt(value, 10);
      return isNaN(parsed) ? 0 : parsed;
    }
    return 0;
  }

  // Method 4: Generate enhanced mock data for development/fallback
  generateMockLiveData(): LiveMatch {
    const teams = [
      'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
      'Kolkata Knight Riders', 'Delhi Capitals', 'Rajasthan Royals',
      'Punjab Kings', 'Sunrisers Hyderabad'
    ];
    
    const venues = [
      'Wankhede Stadium, Mumbai', 'Eden Gardens, Kolkata',
      'M. Chinnaswamy Stadium, Bangalore', 'MA Chidambaram Stadium, Chennai',
      'Arun Jaitley Stadium, Delhi', 'Sawai Mansingh Stadium, Jaipur',
      'PCA Stadium, Mohali', 'Rajiv Gandhi International Stadium, Hyderabad'
    ];
    
    const team1 = teams[Math.floor(Math.random() * teams.length)];
    let team2 = teams[Math.floor(Math.random() * teams.length)];
    while (team2 === team1) {
      team2 = teams[Math.floor(Math.random() * teams.length)];
    }

    const overs = Math.random() * 20;
    const currentScore = Math.floor(Math.random() * 200) + 50;
    const wickets = Math.floor(Math.random() * 8);
    const target = Math.floor(Math.random() * 50) + 150;

    return {
      matchId: `mock-${Date.now()}`,
      team1,
      team2,
      currentScore,
      wickets,
      overs: Math.floor(overs),
      balls: Math.round((overs % 1) * 6),
      target,
      status: 'live',
      venue: venues[Math.floor(Math.random() * venues.length)],
      tossWinner: team1,
      tossDecision: Math.random() > 0.5 ? 'bat' : 'bowl',
      battingTeam: team1,
      bowlingTeam: team2,
      lastUpdated: new Date().toISOString()
    };
  }

  // Main method to get live matches with fallback
  async getLiveMatches(): Promise<LiveMatch[]> {
    try {
      console.log('Fetching live matches with API key...');
      
      // Try to get real IPL matches
      let matches = await this.getCricApiMatches();
      
      console.log(`Found ${matches.length} IPL matches from API`);
      
      // If no real IPL matches found, add mock data for demonstration
      if (matches.length === 0) {
        console.log('No live IPL matches found, generating mock data...');
        matches = [this.generateMockLiveData()];
      }
      
      return matches;
    } catch (error) {
      console.error('Error getting live matches:', error);
      // Fallback to mock data
      return [this.generateMockLiveData()];
    }
  }

  // Get ball-by-ball commentary
  async getBallByBallData(matchId: string): Promise<BallByBallData[]> {
    try {
      const response = await fetch(`${this.baseUrl}/match_info?apikey=${this.apiKey}&id=${matchId}`);
      const data = await response.json();
      
      if (!data.success) {
        console.warn('Ball-by-ball data not available:', data.reason);
        return [];
      }
      
      // Process ball-by-ball data if available
      return data.data?.commentary?.map((ball: any, index: number) => ({
        over: Math.floor(index / 6) + 1,
        ball: (index % 6) + 1,
        runs: ball.runs || 0,
        isWicket: ball.isWicket || false,
        batsman: ball.batsman || '',
        bowler: ball.bowler || '',
        commentary: ball.comm || ''
      })) || [];
    } catch (error) {
      console.error('Error fetching ball-by-ball data:', error);
      return [];
    }
  }

  // Check API status and remaining quota
  async checkApiStatus(): Promise<{ success: boolean; remaining?: number; message: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/currentMatches?apikey=${this.apiKey}&offset=0`);
      const data = await response.json();
      
      if (data.success) {
        return {
          success: true,
          remaining: data.info?.hitsLimit - data.info?.hitsUsed,
          message: `API working. ${data.info?.hitsLimit - data.info?.hitsUsed || 'Unknown'} requests remaining.`
        };
      } else {
        return {
          success: false,
          message: data.reason || 'API request failed'
        };
      }
    } catch (error) {
      return {
        success: false,
        message: `API connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }
}

export const cricketApi = new CricketApiService();