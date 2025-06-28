import { useState, useEffect, useCallback } from 'react';
import { cricketApi, LiveMatch } from '../services/cricketApi';

export function useLiveMatches() {
  const [matches, setMatches] = useState<LiveMatch[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchMatches = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const liveMatches = await cricketApi.getLiveMatches();
      setMatches(liveMatches);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch matches');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMatches();
    
    // Set up polling for real-time updates (every 30 seconds)
    const interval = setInterval(fetchMatches, 30000);
    
    return () => clearInterval(interval);
  }, [fetchMatches]);

  return {
    matches,
    loading,
    error,
    lastUpdated,
    refetch: fetchMatches
  };
}