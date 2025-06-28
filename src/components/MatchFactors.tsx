import React from 'react';
import { BarChart3, Zap, Shield, Target } from 'lucide-react';

interface MatchFactorsProps {
  factors: {
    momentum: number;
    pressure: number;
    form: number;
    conditions: number;
  };
}

export default function MatchFactors({ factors }: MatchFactorsProps) {
  const factorData = [
    { name: 'Momentum', value: factors.momentum, icon: Zap, color: 'bg-yellow-500' },
    { name: 'Pressure Handling', value: factors.pressure, icon: Shield, color: 'bg-red-500' },
    { name: 'Current Form', value: factors.form, icon: Target, color: 'bg-green-500' },
    { name: 'Match Conditions', value: factors.conditions, icon: BarChart3, color: 'bg-blue-500' },
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center space-x-2 mb-6">
        <BarChart3 className="w-5 h-5 text-purple-600" />
        <h3 className="text-lg font-bold text-gray-800">Key Match Factors</h3>
      </div>

      <div className="space-y-4">
        {factorData.map((factor) => {
          const IconComponent = factor.icon;
          return (
            <div key={factor.name} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <IconComponent className="w-4 h-4 text-gray-600" />
                  <span className="text-sm font-medium text-gray-700">{factor.name}</span>
                </div>
                <span className="text-sm font-semibold text-gray-800">{factor.value}/10</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-full rounded-full transition-all duration-500 ${factor.color}`}
                  style={{ width: `${factor.value * 10}%` }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-6 p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg">
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-1">Overall Match Complexity</div>
          <div className="text-2xl font-bold text-purple-600">
            {((factors.momentum + factors.pressure + factors.form + factors.conditions) / 4).toFixed(1)}/10
          </div>
        </div>
      </div>
    </div>
  );
}