// components/ResultsDisplay.jsx
import BillSummary from './BillSummary';
import PredictionChart from './PredictionChart';
import RecommendationsList from './RecommendationsList';

function ResultsDisplay({ results, mode }) {
  return (
    <div className="mt-8 space-y-8">
      <h2 className="text-2xl font-bold text-center text-blue-600">Analysis Results</h2>
      
      {/* User Info */}
      <BillSummary 
        userInfo={results.user_info} 
        currentBill={results.current_bill || {
          kwh_used: results.bill_data?.kwh_used,
          total_bill_amount: results.bill_data?.total_bill_amount
        }}
      />
      
      {/* Predictions */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold mb-4">Future Bill Predictions</h3>
        <PredictionChart predictions={results.predictions} />
      </div>
      
      {/* Display Appliance Breakdown if available */}
      {mode === 'combined' && results.prediction && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4">Appliance Usage Breakdown</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-lg font-medium mb-3">Energy Distribution</h4>
              {/* Appliance breakdown chart would go here */}
              <div className="space-y-2">
                {Object.entries(results.prediction.breakdown).map(([appliance, data]) => (
                  <div key={appliance} className="flex items-center">
                    <div className="w-1/3 text-sm font-medium">
                      {appliance.replace('_', ' ').replace(/^\w/, c => c.toUpperCase())}
                    </div>
                    <div className="w-2/3">
                      <div className="bg-gray-200 h-4 rounded-full overflow-hidden">
                        <div 
                          className="bg-blue-600 h-full rounded-full"
                          style={{ width: `${data.percentage}%` }}
                        />
                      </div>
                      <div className="flex justify-between text-xs mt-1">
                        <span>{data.hours_per_day} hrs/day</span>
                        <span>{data.percentage}%</span>
                        <span>{data.monthly_kwh} kWh</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
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
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* AI Recommendations */}
      <RecommendationsList recommendations={results.ai_recommendations} />
      
      {/* Anomalies (if any) */}
      {results.anomalies && results.anomalies.length > 0 && results.anomalies[0].type !== 'info' && (
        <div className="bg-yellow-50 rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4 text-yellow-800">Anomalies Detected</h3>
          <ul className="space-y-2">
            {results.anomalies.map((anomaly, index) => (
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
    </div>
  );
}

export default ResultsDisplay;