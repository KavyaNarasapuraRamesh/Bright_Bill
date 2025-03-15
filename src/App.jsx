// App.jsx
import { useState } from 'react';
import BillUploader from './components/BillUploader';
import ApplianceSimulator from './components/ApplianceSimulator';
import ResultsDisplay from './components/ResultsDisplay';

function App() {
  const [uploadMode, setUploadMode] = useState('bill-only');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleModeChange = (mode) => {
    setUploadMode(mode);
    setAnalysisResults(null);
  };

  const handleBillUpload = async (formData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setAnalysisResults(data);
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCombinedUpload = async (formData) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/api/combined-prediction', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setAnalysisResults(data);
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <h1 className="text-3xl font-bold text-center text-blue-600 mb-8">
          Electricity Bill Analyzer
        </h1>
        
        {/* Mode Selection */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4">Select Analysis Mode</h2>
          <div className="flex space-x-4">
            <button
              onClick={() => handleModeChange('bill-only')}
              className={`px-4 py-2 rounded-md ${
                uploadMode === 'bill-only' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-800'
              }`}
            >
              Bill Upload Only
            </button>
            <button
              onClick={() => handleModeChange('combined')}
              className={`px-4 py-2 rounded-md ${
                uploadMode === 'combined' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-800'
              }`}
            >
              Bill + Appliance Usage
            </button>
          </div>
        </div>
        
        {/* Uploader Components */}
        {uploadMode === 'bill-only' ? (
          <BillUploader onUpload={handleBillUpload} />
        ) : (
          <ApplianceSimulator onSubmit={handleCombinedUpload} />
        )}
        
        {/* Loading State */}
        {loading && (
          <div className="text-center py-10">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-700 mx-auto"></div>
            <p className="mt-4 text-gray-600">Analyzing your electricity bill...</p>
          </div>
        )}
        
        {/* Error Display */}
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mt-6">
            <strong className="font-bold">Error!</strong>
            <span className="block sm:inline"> {error}</span>
          </div>
        )}
        
        {/* Results Display */}
        {analysisResults && !loading && (
          <ResultsDisplay results={analysisResults} mode={uploadMode} />
        )}
      </div>
    </div>
  );
}

export default App;