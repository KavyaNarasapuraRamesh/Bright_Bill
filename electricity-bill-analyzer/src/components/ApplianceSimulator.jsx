import { useState } from 'react';

function ApplianceSimulator({ onSubmit }) {
  const [file, setFile] = useState(null);
  const [appliances, setAppliances] = useState({
    air_conditioner: 0,
    refrigerator: 24,
    water_heater: 0,
    clothes_dryer: 0,
    washing_machine: 0
  });
  const [household, setHousehold] = useState({
    household_size: 3,
    home_sqft: 1800
  });

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleApplianceChange = (appliance, value) => {
    setAppliances({
      ...appliances,
      [appliance]: parseFloat(value)
    });
  };

  const handleHouseholdChange = (field, value) => {
    setHousehold({
      ...household,
      [field]: parseFloat(value)
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Add appliance usage data
    Object.entries(appliances).forEach(([key, value]) => {
      formData.append(key, value);
    });
    
    // Add household data
    Object.entries(household).forEach(([key, value]) => {
      formData.append(key, value);
    });
    
    onSubmit(formData);
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-8">
      <h2 className="text-xl font-semibold mb-4">Bill & Appliance Usage Analysis</h2>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Bill Upload Section */}
        <div className="p-4 border border-gray-200 rounded-md">
          <h3 className="font-medium mb-2">Upload Your Electricity Bill</h3>
          <input 
            type="file" 
            accept=".pdf" 
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-md file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100"
          />
          {file && <p className="mt-2 text-sm text-green-600">Selected: {file.name}</p>}
        </div>
        
        {/* Appliance Usage Section */}
        <div className="p-4 border border-gray-200 rounded-md">
          <h3 className="font-medium mb-4">Daily Appliance Usage (hours)</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Air Conditioner */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label htmlFor="ac-slider" className="block text-sm font-medium text-gray-700">
                  Air Conditioner
                </label>
                <span className="text-sm text-gray-500">{appliances.air_conditioner} hrs/day</span>
              </div>
              <input 
                type="range" 
                id="ac-slider" 
                min="0" 
                max="24" 
                step="0.5" 
                value={appliances.air_conditioner} 
                onChange={(e) => handleApplianceChange('air_conditioner', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            {/* Refrigerator */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label htmlFor="ref-slider" className="block text-sm font-medium text-gray-700">
                  Refrigerator
                </label>
                <span className="text-sm text-gray-500">{appliances.refrigerator} hrs/day</span>
              </div>
              <input 
                type="range" 
                id="ref-slider" 
                min="0" 
                max="24" 
                step="0.5" 
                value={appliances.refrigerator} 
                onChange={(e) => handleApplianceChange('refrigerator', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            {/* Water Heater */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label htmlFor="wh-slider" className="block text-sm font-medium text-gray-700">
                  Water Heater
                </label>
                <span className="text-sm text-gray-500">{appliances.water_heater} hrs/day</span>
              </div>
              <input 
                type="range" 
                id="wh-slider" 
                min="0" 
                max="8" 
                step="0.5" 
                value={appliances.water_heater} 
                onChange={(e) => handleApplianceChange('water_heater', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            {/* Clothes Dryer */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label htmlFor="cd-slider" className="block text-sm font-medium text-gray-700">
                  Clothes Dryer
                </label>
                <span className="text-sm text-gray-500">{appliances.clothes_dryer} hrs/day</span>
              </div>
              <input 
                type="range" 
                id="cd-slider" 
                min="0" 
                max="5" 
                step="0.5" 
                value={appliances.clothes_dryer} 
                onChange={(e) => handleApplianceChange('clothes_dryer', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            {/* Washing Machine */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label htmlFor="wm-slider" className="block text-sm font-medium text-gray-700">
                  Washing Machine
                </label>
                <span className="text-sm text-gray-500">{appliances.washing_machine} hrs/day</span>
              </div>
              <input 
                type="range" 
                id="wm-slider" 
                min="0" 
                max="5" 
                step="0.5" 
                value={appliances.washing_machine} 
                onChange={(e) => handleApplianceChange('washing_machine', e.target.value)}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        </div>
        
        {/* Household Info Section */}
        <div className="p-4 border border-gray-200 rounded-md">
          <h3 className="font-medium mb-4">Household Information</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="household-size" className="block text-sm font-medium text-gray-700 mb-1">
                Household Size
              </label>
              <input 
                type="number" 
                id="household-size" 
                min="1" 
                max="10" 
                value={household.household_size} 
                onChange={(e) => handleHouseholdChange('household_size', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
            
            <div>
              <label htmlFor="home-sqft" className="block text-sm font-medium text-gray-700 mb-1">
                Home Size (sq ft)
              </label>
              <input 
                type="number" 
                id="home-sqft" 
                min="500" 
                max="5000" 
                step="100" 
                value={household.home_sqft} 
                onChange={(e) => handleHouseholdChange('home_sqft', e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
          </div>
        </div>
        
        <button 
          type="submit" 
          className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed"
          disabled={!file}
        >
          Analyze Bill & Appliance Usage
        </button>
      </form>
    </div>
  );
}

export default ApplianceSimulator;