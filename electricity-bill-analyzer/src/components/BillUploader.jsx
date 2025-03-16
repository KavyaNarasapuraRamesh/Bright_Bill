import { useState } from 'react';

function BillUploader({ onUpload, disabled = false }) {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [printEnabled, setPrintEnabled] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('future_months', '3');
    
    // Add print_enabled parameter
    formData.append('print_enabled', printEnabled);
    
    console.log('Submitting file:', file.name);
    console.log('Print enabled:', printEnabled);
    
    onUpload(formData);
  };

  return (
    <div className={`bg-white rounded-lg shadow-md p-6 mb-8 ${disabled ? 'opacity-60' : ''}`}>
      <h2 className="text-xl font-semibold mb-4">Upload Your Electricity Bill</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div 
          className={`border-2 border-dashed rounded-lg p-8 text-center ${
            disabled ? 'bg-gray-100 cursor-not-allowed' : 
            dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
          }`}
          onDragEnter={disabled ? null : handleDrag}
          onDragLeave={disabled ? null : handleDrag}
          onDragOver={disabled ? null : handleDrag}
          onDrop={disabled ? null : handleDrop}
        >
          {file ? (
            <div className="text-green-600">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p>{file.name}</p>
            </div>
          ) : (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto text-gray-400 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="text-gray-500">
                {disabled 
                  ? "File upload disabled - backend unavailable" 
                  : "Drag & drop your bill PDF here, or click to select"}
              </p>
            </>
          )}
          
          <input 
            type="file" 
            className="hidden" 
            accept=".pdf,.jpg,.jpeg,.png" 
            onChange={handleChange}
            id="bill-file-input"
            disabled={disabled}
          />
          <label 
            htmlFor="bill-file-input" 
            className={`mt-4 inline-block px-4 py-2 bg-blue-600 text-white rounded-md ${
              disabled 
                ? 'opacity-50 cursor-not-allowed' 
                : 'cursor-pointer hover:bg-blue-700'
            }`}
          >
            Select File
          </label>
        </div>
        
        {/* Print option */}
        <div className="flex items-center">
          <input
            id="print-option"
            type="checkbox"
            className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            checked={printEnabled}
            onChange={() => setPrintEnabled(!printEnabled)}
            disabled={disabled}
          />
          <label htmlFor="print-option" className="ml-2 text-sm text-gray-700">
            Enable report printing
          </label>
        </div>
        
        <button 
          type="submit" 
          className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed"
          disabled={!file || disabled}
        >
          Analyze Bill
        </button>
      </form>
    </div>
  );
}

export default BillUploader;