// components/PredictionChart.jsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function PredictionChart({ predictions }) {
  // Format data for the chart
  const chartData = predictions.map(prediction => ({
    date: new Date(prediction.prediction_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }),
    kwh: prediction.predicted_kwh,
    cost: prediction.total_bill_amount,
    utility: prediction.utility_charges,
    supplier: prediction.supplier_charges
  }));

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
              <p className="flex justify-between">
                <span className="text-gray-600">Utility Charges:</span>
                <span className="font-medium">${month.utility}</span>
              </p>
              <p className="flex justify-between">
                <span className="text-gray-600">Supplier Charges:</span>
                <span className="font-medium">${month.supplier}</span>
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PredictionChart;