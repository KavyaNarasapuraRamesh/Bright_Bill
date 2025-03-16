import { useState, useEffect } from 'react';
import BillUploader from './components/BillUploader';
import ApplianceSimulator from './components/ApplianceSimulator';
import ResultsDisplay from './components/ResultsDisplay';

// Backend API URL
const API_URL = 'http://127.0.0.1:8000'; // Use 127.0.0.1 instead of localhost

function App() {
  const [uploadMode, setUploadMode] = useState('bill-only');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');

  // Check if backend is running
  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    setBackendStatus('checking');
    
    try {
      console.log('Checking backend connection...');
      const response = await fetch(`${API_URL}/`);
      
      if (response.ok) {
        const data = await response.json();
        console.log('Backend response:', data);
        setBackendStatus('connected');
      } else {
        console.error('Backend returned error:', response.status);
        setBackendStatus('error');
      }
    } catch (err) {
      console.error('Backend connection error:', err);
      setBackendStatus('disconnected');
    }
  };

  const handleModeChange = (mode) => {
    setUploadMode(mode);
    setAnalysisResults(null);
  };

  const handleBillUpload = async (formData) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('Sending request to:', `${API_URL}/api/upload`);
      
      const response = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Error: ${response.status} - ${errorText}`);
      }
      
      const data = await response.json();
      console.log('Received data:', data);
      setAnalysisResults(data);
    } catch (err) {
      console.error('Upload error:', err);
      setError(`${err.message}. Please ensure the backend server is running.`);
      
      // Check backend connection again
      checkBackendConnection();
    } finally {
      setLoading(false);
    }
  };

  const handleCombinedUpload = async (formData) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('Sending request to:', `${API_URL}/api/combined-prediction`);
      
      const response = await fetch(`${API_URL}/api/combined-prediction`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Error: ${response.status} - ${errorText}`);
      }
      
      const data = await response.json();
      console.log('Received data:', data);
      setAnalysisResults(data);
    } catch (err) {
      console.error('Combined upload error:', err);
      setError(`${err.message}. Please ensure the backend server is running.`);
      
      // Check backend connection again
      checkBackendConnection();
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
        
        {/* Backend Status */}
        <div className="bg-white rounded-lg shadow-md p-4 mb-6 flex justify-between items-center">
          <div className="flex items-center">
            <span className="mr-2">Backend Status:</span>
            {backendStatus === 'checking' && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                Checking...
              </span>
            )}
            {backendStatus === 'connected' && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Connected
              </span>
            )}
            {(backendStatus === 'disconnected' || backendStatus === 'error') && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                Disconnected
              </span>
            )}
            {(backendStatus === 'disconnected' || backendStatus === 'error') && (
              <button 
                onClick={checkBackendConnection}
                className="ml-2 px-2 py-1 text-xs bg-blue-600 text-white rounded"
              >
                Retry
              </button>
            )}
          </div>
          
          <div>
            <span className="text-xs text-gray-500">API: {API_URL}</span>
          </div>
        </div>
        
        {(backendStatus === 'disconnected' || backendStatus === 'error') && (
          <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 mb-6">
            <p className="font-bold">Backend Server Not Available</p>
            <p>Please make sure the FastAPI backend server is running at {API_URL}</p>
            <p className="mt-2 text-sm">
              Run this command in your terminal: <code className="bg-gray-200 px-1">python simple_api_stub.py</code>
            </p>
          </div>
        )}
        
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
          <BillUploader 
            onUpload={handleBillUpload} 
            disabled={backendStatus !== 'connected'} 
          />
        ) : (
          <ApplianceSimulator 
            onSubmit={handleCombinedUpload}
            disabled={backendStatus !== 'connected'}
          />
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