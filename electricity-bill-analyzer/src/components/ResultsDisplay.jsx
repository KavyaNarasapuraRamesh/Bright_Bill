import BillSummary from './BillSummary';
import PredictionChart from './PredictionChart';
import AppliancePieChart from './AppliancePieChart';
import CompactRecommendations from './CompactRecommendations';

function ResultsDisplay({ results, mode }) {
  // Add safety checks and data preprocessing
  const predictions = results.predictions || [];
  
  return (
    <div className="mt-8 space-y-8">
      <h2 className="text-2xl font-bold text-center text-blue-600">Analysis Results</h2>
      
      {/* User Info */}
      <BillSummary 
        userInfo={results.user_info} 
        currentBill={results.current_bill || results.bill_data || {}}
      />
      
      {/* Different visualizations based on mode */}
      {mode === 'bill-only' && predictions.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4">Future Bill Predictions</h3>
          <PredictionChart predictions={predictions} />
        </div>
      )}
      
      {/* Display Appliance Breakdown with pie chart if available */}
      {mode === 'combined' && results.prediction && results.prediction.breakdown && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4">Appliance Usage Breakdown</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-lg font-medium mb-3">Energy Distribution</h4>
              {/* Use pie chart component for appliance breakdown */}
              <AppliancePieChart breakdown={results.prediction.breakdown} />
            </div>
            
            <div>
              <h4 className="text-lg font-medium mb-3">Summary</h4>
              <div className="space-y-4">
                <div className="bg-blue-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">Predicted Monthly Usage</p>
                  <p className="text-2xl font-bold text-blue-700">{results.prediction.total_kwh} kWh</p>
                </div>
                
                <div className="bg-green-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">Estimated Monthly Cost</p>
                  <p className="text-2xl font-bold text-green-700">${results.prediction.estimated_cost}</p>
                </div>
                
                {/* Add comparison to current usage if available */}
                {results.current_bill && results.current_bill.kwh_used && (
                  <div className="bg-purple-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">Compared to Current Usage</p>
                    <p className="text-2xl font-bold text-purple-700">
                      {calculatePercentChange(results.current_bill.kwh_used, results.prediction.total_kwh)}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* AI Recommendations - Using new compact version */}
      {results.ai_recommendations && results.ai_recommendations.length > 0 && (
        <CompactRecommendations recommendations={results.ai_recommendations} />
      )}
      
      {/* Anomalies (if any) */}
      {results.anomalies && results.anomalies.length > 0 && results.anomalies[0].type !== 'info' && (
        <div className="bg-yellow-50 rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4 text-yellow-800">Anomalies Detected</h3>
          <ul className="space-y-2">
            {results.anomalies
              .filter(anomaly => anomaly.type !== 'info')
              .map((anomaly, index) => (
                <li key={index} className="flex items-start">
                  <span className={`inline-block w-2 h-2 mt-2 mr-2 rounded-full ${
                    anomaly.severity === 'high' ? 'bg-red-500' : 'bg-yellow-500'
                  }`}></span>
                  <span>{anomaly.description}</span>
                </li>
              ))}
          </ul>
        </div>
      )}
      
      {/* Print option indication if enabled */}
      {results.print_enabled && (
        <div className="bg-gray-100 p-4 rounded-lg text-center">
          <p className="text-gray-600">Report will be printed automatically</p>
        </div>
      )}
    </div>
  );
}

// Helper function to calculate and format percent change
function calculatePercentChange(oldValue, newValue) {
  if (!oldValue || !newValue) return "N/A";
  
  const percentChange = ((newValue - oldValue) / oldValue) * 100;
  const formattedChange = percentChange.toFixed(1);
  
  if (percentChange > 0) {
    return `↑ ${formattedChange}%`;
  } else if (percentChange < 0) {
    return `↓ ${Math.abs(formattedChange)}%`;
  } else {
    return "No change";
  }
}

export default ResultsDisplay;