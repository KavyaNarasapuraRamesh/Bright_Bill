import React, { useState } from 'react';

// Badge icon mapping
const BADGE_ICONS = {
  'seedling': (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h.01M12 12h.01M19 12h.01M6 12a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0z" />
    </svg>
  ),
  'leaf': (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  'tree': (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  'fire': (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.879 16.121A3 3 0 1012.015 11L11 14H9c0 .768.293 1.536.879 2.121z" />
    </svg>
  ),
  'bolt': (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
  'star': (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
    </svg>
  ),
  'clock': (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  'rocket': (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.59 14.37a6 6 0 01-5.84 7.38v-4.8m5.84-2.58a14.98 14.98 0 006.16-12.12A14.98 14.98 0 009.631 8.41m5.96 5.96a14.926 14.926 0 01-5.841 2.58m-.119-8.54a6 6 0 00-7.381 5.84h4.8m2.581-5.84a14.927 14.927 0 00-2.58 5.84m2.699 2.7c-.103.021-.207.041-.311.06a15.09 15.09 0 01-2.448-2.448 14.9 14.9 0 01.06-.312m-2.24 2.39a4.493 4.493 0 00-1.757 4.306 4.493 4.493 0 004.306-1.758M16.5 9a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0z" />
    </svg>
  )
};

function GamificationCard({ gamification }) {
  const [showAllAchievements, setShowAllAchievements] = useState(false);
  
  // Debug log
  console.log("Gamification data received:", gamification);
  
  if (!gamification) {
    console.log("No gamification data available");
    return null;
  }
  
  // Extract gamification properties with fallbacks for any missing data
  const { 
    level = { name: "Energy Novice", progress: 0, points_to_next_level: 50, next_level: "Energy Apprentice" }, 
    total_points = 0, 
    reduction_streak = 0
  } = gamification;
  
  // Handle both possible achievement data structures
  let new_achievements = [];
  let recent_achievements = [];
  
  if (Array.isArray(gamification.new_achievements)) {
    new_achievements = gamification.new_achievements;
  }
  
  if (Array.isArray(gamification.recent_achievements)) {
    recent_achievements = gamification.recent_achievements;
  } else if (Array.isArray(gamification.achievements)) {
    // If we have a simple array of achievements, use that for both
    new_achievements = gamification.achievements || [];
    recent_achievements = gamification.achievements || [];
  }
  
  // Determine which achievements to show
  const displayAchievements = showAllAchievements 
    ? recent_achievements 
    : new_achievements.length > 0 
      ? new_achievements 
      : recent_achievements.slice(0, 3);

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h3 className="text-xl font-semibold mb-4 flex items-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
        Your Energy Profile
      </h3>
      
      {/* Level and Points */}
      <div className="bg-green-50 rounded-lg p-4 mb-6">
        <div className="flex flex-col md:flex-row md:justify-between md:items-center">
          <div className="mb-4 md:mb-0">
            <h4 className="text-sm font-medium text-gray-600">Energy Level</h4>
            <p className="text-2xl font-bold text-green-700">{level.name}</p>
            <p className="text-sm text-gray-500">
              {level.points_to_next_level > 0 
                ? `${level.points_to_next_level} points to next level: ${level.next_level}`
                : "Maximum level reached!"}
            </p>
          </div>
          
          <div className="text-center">
            <div className="inline-flex items-center justify-center p-4 bg-green-100 rounded-full w-20 h-20">
              <span className="text-2xl font-bold text-green-700">{total_points}</span>
            </div>
            <p className="text-sm font-medium text-gray-600 mt-1">Total Points</p>
          </div>
        </div>
        
        {/* Progress bar */}
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className="bg-green-600 h-2.5 rounded-full" 
              style={{ width: `${level.progress}%` }}
            ></div>
          </div>
          <div className="flex justify-between mt-1 text-xs text-gray-500">
            <span>{level.name}</span>
            <span>{level.next_level || "Max Level"}</span>
          </div>
        </div>
      </div>
      
      {/* Achievements Section */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-3">
          <h4 className="font-medium text-gray-700">
            {new_achievements.length > 0 ? 'New Achievements!' : 'Recent Achievements'}
          </h4>
          
          {recent_achievements.length > 3 && (
            <button 
              onClick={() => setShowAllAchievements(!showAllAchievements)}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              {showAllAchievements ? 'Show Less' : 'View All'}
            </button>
          )}
        </div>
        
        {displayAchievements.length > 0 ? (
          <div className="space-y-3">
            {displayAchievements.map((achievement, index) => (
              <div key={index} className="flex items-start p-3 bg-gray-50 rounded-lg">
                <div className="flex-shrink-0 p-2 bg-green-500 text-white rounded-full mr-3">
                  {BADGE_ICONS[achievement.badge] || (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  )}
                </div>
                <div className="flex-1">
                  <h5 className="font-medium">{achievement.title}</h5>
                  <p className="text-sm text-gray-600">{achievement.description}</p>
                </div>
                <div className="flex-shrink-0 text-right">
                  <span className="inline-block px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                    +{achievement.points} pts
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-4 bg-gray-50 rounded-lg text-center text-gray-500">
            No achievements yet. Continue using the app to earn points!
          </div>
        )}
      </div>
      
      {/* Reduction Streak */}
      {reduction_streak > 0 && (
        <div className="p-4 bg-orange-50 rounded-lg">
          <div className="flex items-center">
            <div className="flex-shrink-0 p-2 bg-orange-500 text-white rounded-full mr-3">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.879 16.121A3 3 0 1012.015 11L11 14H9c0 .768.293 1.536.879 2.121z" />
              </svg>
            </div>
            <div>
              <h5 className="font-medium">Energy Saving Streak!</h5>
              <p className="text-sm text-gray-600">
                You've reduced your energy usage for {reduction_streak} {reduction_streak === 1 ? 'month' : 'months'} in a row!
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default GamificationCard;