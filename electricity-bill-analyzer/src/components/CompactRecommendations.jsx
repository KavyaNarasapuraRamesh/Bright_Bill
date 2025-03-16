import React, { useState } from 'react';

// Helper function to get icon based on recommendation category
const getRecommendationIcon = (title) => {
  const titleLower = title.toLowerCase();
  
  if (titleLower.includes('air conditioner') || titleLower.includes('cooling')) {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    );
  } else if (titleLower.includes('refrigerator')) {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
      </svg>
    );
  } else if (titleLower.includes('thermostat') || titleLower.includes('temperature')) {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
      </svg>
    );
  } else if (titleLower.includes('phantom') || titleLower.includes('standby')) {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    );
  } else if (titleLower.includes('peak') || titleLower.includes('time')) {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    );
  } else if (titleLower.includes('water') || titleLower.includes('shower')) {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    );
  } else if (titleLower.includes('light') || titleLower.includes('lamp')) {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    );
  } else {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    );
  }
};

function CompactRecommendations({ recommendations }) {
  const [feedback, setFeedback] = useState({});
  const [implementStatus, setImplementStatus] = useState({});

  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  // Keep only 3-4 most relevant recommendations
  const topRecommendations = recommendations.slice(0, 4);

  const handleFeedback = (id, value) => {
    setFeedback({ ...feedback, [id]: value });
    
    // Here you could implement an API call to send feedback
    // This would be the point to integrate with a more sophisticated NLP system
    console.log(`Recommendation ${id} feedback: ${value}`);
  };

  const handleImplementStatus = (id) => {
    setImplementStatus({ 
      ...implementStatus, 
      [id]: !implementStatus[id] 
    });
    
    // Here you could track which recommendations users plan to implement
    console.log(`Recommendation ${id} implementation status: ${!implementStatus[id]}`);
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-xl font-semibold mb-4">Personalized Recommendations</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {topRecommendations.map((rec, index) => (
          <div key={index} className="bg-blue-50 rounded-lg p-4 transition-all hover:shadow-md">
            <div className="flex items-start space-x-3">
              <div className="p-2 bg-blue-500 text-white rounded-full">
                {getRecommendationIcon(rec.title)}
              </div>
              
              <div className="flex-1">
                <h4 className="font-medium text-blue-800">{rec.title}</h4>
                <p className="text-sm text-gray-600 mt-1 line-clamp-2">
                  {rec.description}
                </p>
                
                <div className="mt-3 flex items-center justify-between">
                  {/* Will implement checkbox */}
                  <label className="flex items-center text-xs text-gray-500 cursor-pointer">
                    <input
                      type="checkbox"
                      className="mr-1 h-4 w-4 text-blue-600 rounded focus:ring-blue-500"
                      checked={implementStatus[index] || false}
                      onChange={() => handleImplementStatus(index)}
                    />
                    I'll implement this
                  </label>
                  
                  {/* Simple sentiment feedback */}
                  <div className="flex space-x-1">
                    <button 
                      onClick={() => handleFeedback(index, 'helpful')}
                      className={`p-1 rounded-full ${feedback[index] === 'helpful' ? 'bg-green-100 text-green-600' : 'text-gray-400 hover:text-green-600'}`}
                      title="This was helpful"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                      </svg>
                    </button>
                    <button 
                      onClick={() => handleFeedback(index, 'not helpful')}
                      className={`p-1 rounded-full ${feedback[index] === 'not helpful' ? 'bg-red-100 text-red-600' : 'text-gray-400 hover:text-red-600'}`}
                      title="Not helpful"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default CompactRecommendations;