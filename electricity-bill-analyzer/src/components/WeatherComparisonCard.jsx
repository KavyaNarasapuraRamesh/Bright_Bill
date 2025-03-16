import React from 'react';

function WeatherComparisonCard({ comparison }) {
  if (!comparison) {
    return null;
  }

  const getRawDiffColor = () => {
    const diff = comparison.raw_diff;
    return diff < 0 ? 'text-green-600' : diff > 0 ? 'text-red-600' : 'text-gray-700';
  };

  const getAdjustedDiffColor = () => {
    const diff = comparison.adjusted_diff;
    return diff < 0 ? 'text-green-600' : diff > 0 ? 'text-red-600' : 'text-gray-700';
  };

  // Calculate percentage of change due to weather
  const weatherContribution = Math.abs(
    comparison.weather_impact / comparison.raw_diff * 100
  );
  
  // Determine if the weather helped or hurt
  const weatherEffect = comparison.weather_impact < 0 ? 'decreased' : 'increased';

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h3 className="text-xl font-semibold mb-4 flex items-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
        </svg>
        Weather-Adjusted Usage Comparison
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Current Usage</p>
          <p className="text-2xl font-bold text-blue-700">{comparison.current_usage} kWh</p>
          <p className="text-xs text-gray-500">at {comparison.current_temperature}°F avg. temp</p>
        </div>
        
        <div className="bg-blue-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Previous Usage</p>
          <p className="text-2xl font-bold text-blue-700">{comparison.previous_usage} kWh</p>
          <p className="text-xs text-gray-500">at {comparison.previous_temperature}°F avg. temp</p>
        </div>
        
        <div className="bg-blue-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">Weather-Adjusted Previous</p>
          <p className="text-2xl font-bold text-blue-700">{comparison.weather_adjusted_previous} kWh</p>
          <p className="text-xs text-gray-500">if weather matched current month</p>
        </div>
      </div>
      
      <div className="border-t border-gray-200 pt-4 mb-6">
        <div className="flex flex-col md:flex-row md:justify-between">
          <div className="mb-4 md:mb-0">
            <h4 className="font-medium text-gray-700 mb-2">Actual Change</h4>
            <p className={`text-xl font-bold ${getRawDiffColor()}`}>
              {comparison.raw_diff < 0 ? '↓' : '↑'} {Math.abs(comparison.raw_diff)} kWh ({Math.abs(comparison.raw_diff_percent)}%)
            </p>
            <p className="text-sm text-gray-600">Without adjusting for weather</p>
          </div>
          
          <div className="mb-4 md:mb-0">
            <h4 className="font-medium text-gray-700 mb-2">Weather-Adjusted Change</h4>
            <p className={`text-xl font-bold ${getAdjustedDiffColor()}`}>
              {comparison.adjusted_diff < 0 ? '↓' : '↑'} {Math.abs(comparison.adjusted_diff)} kWh ({Math.abs(comparison.adjusted_diff_percent)}%)
            </p>
            <p className="text-sm text-gray-600">Your true usage change</p>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Weather Impact</h4>
            <p className="text-xl font-bold text-purple-700">
              {Math.abs(comparison.weather_impact)} kWh ({Math.abs(comparison.weather_impact_percent)}%)
            </p>
            <p className="text-sm text-gray-600">Weather {weatherEffect} your usage</p>
          </div>
        </div>
      </div>
      
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-medium text-gray-700 mb-2">What This Means</h4>
        <p className="text-gray-600">{comparison.explanation}</p>
        
        {Math.abs(comparison.weather_impact) > 20 && (
          <div className="mt-2 text-sm">
            <p className="text-blue-700 font-medium">
              Weather was responsible for about {Math.round(weatherContribution)}% of your usage change.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default WeatherComparisonCard;