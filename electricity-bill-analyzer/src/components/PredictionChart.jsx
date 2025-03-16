import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function PredictionChart({ predictions }) {
  // Add safety check for missing predictions
  if (!predictions || !Array.isArray(predictions) || predictions.length === 0) {
    return (
      <div className="bg-gray-50 p-4 rounded-lg text-center">
        <p className="text-gray-500">No prediction data available</p>
      </div>
    );
  }

  // Format data for the chart with safer property access
  const chartData = predictions.map(prediction => {
    // Use optional chaining and default values to handle different data structures
    return {
      date: new Date(prediction.prediction_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }),
      kwh: prediction.predicted_kwh || 0,
      cost: prediction.total_bill_amount || 0,
      utility: prediction.utility_charges || 0,
      supplier: prediction.supplier_charges || 0
    };
  });

  return (
    <div className="space-y-6">
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
            <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
            <Tooltip />
            <Legend />
            <Line 
              yAxisId="left"
              type="monotone" 
              dataKey="kwh" 
              name="Electricity Usage (kWh)" 
              stroke="#8884d8" 
              activeDot={{ r: 8 }} 
            />
            <Line 
              yAxisId="right"
              type="monotone" 
              dataKey="cost" 
              name="Bill Amount ($)" 
              stroke="#82ca9d" 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {chartData.map((month, index) => (
          <div key={index} className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-semibold text-blue-700">{month.date}</h4>
            <div className="mt-2 space-y-1">
              <p className="flex justify-between">
                <span className="text-gray-600">Usage:</span>
                <span className="font-medium">{month.kwh} kWh</span>
              </p>
              <p className="flex justify-between">
                <span className="text-gray-600">Bill Amount:</span>
                <span className="font-medium">${month.cost}</span>
              </p>
              {month.utility > 0 && (
                <p className="flex justify-between">
                  <span className="text-gray-600">Utility Charges:</span>
                  <span className="font-medium">${month.utility}</span>
                </p>
              )}
              {month.supplier > 0 && (
                <p className="flex justify-between">
                  <span className="text-gray-600">Supplier Charges:</span>
                  <span className="font-medium">${month.supplier}</span>
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PredictionChart;