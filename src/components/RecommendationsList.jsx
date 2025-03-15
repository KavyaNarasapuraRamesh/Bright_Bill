// components/RecommendationsList.jsx
function RecommendationsList({ recommendations }) {
    if (!recommendations || recommendations.length === 0) {
      return null;
    }
  
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold mb-4">Personalized Recommendations</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {recommendations.map((recommendation, index) => (
            <div 
              key={index} 
              className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-500"
            >
              <h4 className="font-medium text-blue-800 mb-2">{recommendation.title}</h4>
              <p className="text-gray-700">{recommendation.description}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }
  
  export default RecommendationsList;