import React from 'react';
import { Shield, Star } from 'lucide-react';

interface Team {
  id: string;
  name: string;
  shortName: string;
  colors: string[];
  primaryColor: string;
}

const teams: Team[] = [
  { id: 'csk', name: 'Chennai Super Kings', shortName: 'CSK', colors: ['from-yellow-400', 'to-yellow-600'], primaryColor: 'bg-yellow-500' },
  { id: 'mi', name: 'Mumbai Indians', shortName: 'MI', colors: ['from-blue-500', 'to-blue-700'], primaryColor: 'bg-blue-600' },
  { id: 'rcb', name: 'Royal Challengers Bangalore', shortName: 'RCB', colors: ['from-red-500', 'to-red-700'], primaryColor: 'bg-red-600' },
  { id: 'kkr', name: 'Kolkata Knight Riders', shortName: 'KKR', colors: ['from-purple-500', 'to-purple-700'], primaryColor: 'bg-purple-600' },
  { id: 'dc', name: 'Delhi Capitals', shortName: 'DC', colors: ['from-blue-400', 'to-blue-600'], primaryColor: 'bg-blue-500' },
  { id: 'rr', name: 'Rajasthan Royals', shortName: 'RR', colors: ['from-pink-500', 'to-pink-700'], primaryColor: 'bg-pink-600' },
  { id: 'pbks', name: 'Punjab Kings', shortName: 'PBKS', colors: ['from-red-400', 'to-red-600'], primaryColor: 'bg-red-500' },
  { id: 'srh', name: 'Sunrisers Hyderabad', shortName: 'SRH', colors: ['from-orange-500', 'to-orange-700'], primaryColor: 'bg-orange-600' },
];

interface TeamSelectorProps {
  selectedTeam: Team | null;
  onTeamSelect: (team: Team) => void;
  label: string;
}

export default function TeamSelector({ selectedTeam, onTeamSelect, label }: TeamSelectorProps) {
  return (
    <div className="space-y-3">
      <label className="block text-sm font-semibold text-gray-700">{label}</label>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {teams.map((team) => (
          <button
            key={team.id}
            onClick={() => onTeamSelect(team)}
            className={`
              relative p-4 rounded-xl border-2 transition-all duration-300 hover:scale-105
              ${selectedTeam?.id === team.id 
                ? 'border-gray-800 shadow-lg' 
                : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
              }
            `}
          >
            <div className={`absolute inset-0 rounded-xl bg-gradient-to-br ${team.colors.join(' ')} opacity-10`}></div>
            <div className="relative flex flex-col items-center space-y-2">
              <div className={`w-8 h-8 rounded-full ${team.primaryColor} flex items-center justify-center`}>
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div className="text-center">
                <div className="font-bold text-sm text-gray-800">{team.shortName}</div>
                <div className="text-xs text-gray-600 truncate">{team.name}</div>
              </div>
              {selectedTeam?.id === team.id && (
                <Star className="w-4 h-4 text-yellow-500 fill-current" />
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

export type { Team };