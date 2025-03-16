import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

function AppliancePieChart({ breakdown }) {
  // If no breakdown data, show empty state
  if (!breakdown || Object.keys(breakdown).length === 0) {
    return (
      <div className="bg-gray-100 p-4 rounded-lg text-center">
        <p className="text-gray-500">No appliance usage data available</p>
      </div>
    );
  }

  // Transform breakdown data for the pie chart
  const pieData = Object.entries(breakdown).map(([appliance, data]) => ({
    name: appliance
      .replace('_', ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' '),
    value: data.percentage,
    kwh: data.monthly_kwh,
    hours: data.hours_per_day
  }))
  .filter(item => item.value > 0) // Only show appliances with usage
  .sort((a, b) => b.value - a.value); // Sort by percentage (descending)

  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, index }) => {
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    // Only show percentage for slices that are big enough to fit text
    return percent > 0.05 ? (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor="middle" 
        dominantBaseline="central"
        fontSize={12}
        fontWeight="bold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    ) : null;
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white shadow-md rounded p-2 text-sm">
          <p className="font-medium">{data.name}</p>
          <p>{data.value}% of total usage</p>
          <p>{data.kwh} kWh / month</p>
          <p>{data.hours} hours/day</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="mt-4">
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={renderCustomizedLabel}
              outerRadius={80}
              innerRadius={30}
              fill="#8884d8"
              dataKey="value"
            >
              {pieData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              layout="vertical" 
              verticalAlign="middle" 
              align="right"
              formatter={(value, entry, index) => (
                <span className="text-sm">{value}</span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-4">
        {pieData.map((item, index) => (
          <div 
            key={index} 
            className="rounded-lg p-3 text-white"
            style={{ backgroundColor: COLORS[index % COLORS.length] }}
          >
            <p className="font-medium">{item.name}</p>
            <div className="flex justify-between items-center mt-1">
              <span>{item.hours} hrs/day</span>
              <span>{item.kwh} kWh</span>
              <span>{item.value}%</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default AppliancePieChart;