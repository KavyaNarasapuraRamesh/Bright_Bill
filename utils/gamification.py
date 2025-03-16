"""
Gamification utilities for electricity bill analysis.
This module provides functions for adding gamification elements
to the electricity bill analysis results.
"""

import json
import os
from datetime import datetime

# Define achievement thresholds and rewards
ACHIEVEMENTS = {
    # Usage reduction achievements
    'usage_reduction_5': {
        'title': 'Energy Saver',
        'description': 'Reduced your energy usage by 5% compared to last month',
        'points': 10,
        'badge': 'seedling'
    },
    'usage_reduction_10': {
        'title': 'Efficiency Expert',
        'description': 'Reduced your energy usage by 10% compared to last month',
        'points': 25,
        'badge': 'leaf'
    },
    'usage_reduction_20': {
        'title': 'Conservation Champion',
        'description': 'Reduced your energy usage by 20% compared to last month',
        'points': 50,
        'badge': 'tree'
    },
    
    # Streak achievements
    'reduction_streak_2': {
        'title': 'Consistent Conserver',
        'description': 'Reduced usage for 2 months in a row',
        'points': 15,
        'badge': 'fire'
    },
    'reduction_streak_3': {
        'title': 'Sustainable Superstar',
        'description': 'Reduced usage for 3 months in a row',
        'points': 30,
        'badge': 'bolt'
    },
    
    # Special achievements
    'below_average': {
        'title': 'Better Than Average',
        'description': 'Your usage is lower than similar households',
        'points': 20,
        'badge': 'star'
    },
    'peak_reduction': {
        'title': 'Peak Reducer',
        'description': 'Successfully reduced usage during peak hours',
        'points': 15,
        'badge': 'clock'
    },
    'first_analysis': {
        'title': 'Energy Explorer',
        'description': 'Analyzed your first electricity bill',
        'points': 5,
        'badge': 'rocket'
    }
}

def load_user_gamification_data(account_number):
    """
    Load gamification data for a user
    
    Args:
        account_number: The user's account number
        
    Returns:
        Dictionary with user's gamification data
    """
    # Create directory if it doesn't exist
    os.makedirs('data/gamification', exist_ok=True)
    
    # Try to load existing data for this user
    file_path = f'data/gamification/{account_number}.json'
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    
    # Return default data if no existing data found
    return {
        'account_number': account_number,
        'total_points': 0,
        'achievements': [],
        'history': [],
        'reduction_streak': 0,
        'last_bill_kwh': None,
        'last_bill_date': None
    }

def save_user_gamification_data(user_data):
    """
    Save gamification data for a user
    
    Args:
        user_data: Dictionary with user's gamification data
    """
    os.makedirs('data/gamification', exist_ok=True)
    
    account_number = user_data.get('account_number')
    if not account_number:
        return False
    
    file_path = f'data/gamification/{account_number}.json'
    with open(file_path, 'w') as f:
        json.dump(user_data, f, indent=2)
    
    return True

def check_achievements(current_bill, previous_bill, user_data):
    """
    Check for new achievements based on bill comparison
    
    Args:
        current_bill: Dictionary with current bill data
        previous_bill: Dictionary with previous bill data (can be None)
        user_data: User's existing gamification data
        
    Returns:
        Tuple of (updated user data, list of new achievements)
    """
    new_achievements = []
    
    # Get key values
    current_usage = current_bill.get('kwh_used', 0)
    account_number = current_bill.get('account_number', 'unknown')
    
    # Make sure user data has correct account number
    user_data['account_number'] = account_number
    
    # Handle first bill case
    if 'first_analysis' not in [a['id'] for a in user_data['achievements']]:
        achievement = ACHIEVEMENTS['first_analysis'].copy()
        achievement['id'] = 'first_analysis'
        achievement['date'] = datetime.now().strftime('%Y-%m-%d')
        
        user_data['achievements'].append(achievement)
        user_data['total_points'] += achievement['points']
        new_achievements.append(achievement)
    
    # If we have previous bill data, check for reduction achievements
    if previous_bill and 'kwh_used' in previous_bill:
        previous_usage = previous_bill.get('kwh_used', 0)
        
        # Calculate percent reduction
        if previous_usage > 0:
            reduction_percent = (previous_usage - current_usage) / previous_usage * 100
            
            # Check for usage reduction achievements
            if reduction_percent >= 20 and 'usage_reduction_20' not in [a['id'] for a in user_data['achievements']]:
                achievement = ACHIEVEMENTS['usage_reduction_20'].copy()
                achievement['id'] = 'usage_reduction_20'
                achievement['date'] = datetime.now().strftime('%Y-%m-%d')
                
                user_data['achievements'].append(achievement)
                user_data['total_points'] += achievement['points']
                new_achievements.append(achievement)
                
            elif reduction_percent >= 10 and 'usage_reduction_10' not in [a['id'] for a in user_data['achievements']]:
                achievement = ACHIEVEMENTS['usage_reduction_10'].copy()
                achievement['id'] = 'usage_reduction_10'
                achievement['date'] = datetime.now().strftime('%Y-%m-%d')
                
                user_data['achievements'].append(achievement)
                user_data['total_points'] += achievement['points']
                new_achievements.append(achievement)
                
            elif reduction_percent >= 5 and 'usage_reduction_5' not in [a['id'] for a in user_data['achievements']]:
                achievement = ACHIEVEMENTS['usage_reduction_5'].copy()
                achievement['id'] = 'usage_reduction_5'
                achievement['date'] = datetime.now().strftime('%Y-%m-%d')
                
                user_data['achievements'].append(achievement)
                user_data['total_points'] += achievement['points']
                new_achievements.append(achievement)
            
            # Update reduction streak
            if reduction_percent > 0:
                user_data['reduction_streak'] += 1
            else:
                user_data['reduction_streak'] = 0
            
            # Check for streak achievements
            if user_data['reduction_streak'] >= 3 and 'reduction_streak_3' not in [a['id'] for a in user_data['achievements']]:
                achievement = ACHIEVEMENTS['reduction_streak_3'].copy()
                achievement['id'] = 'reduction_streak_3'
                achievement['date'] = datetime.now().strftime('%Y-%m-%d')
                
                user_data['achievements'].append(achievement)
                user_data['total_points'] += achievement['points']
                new_achievements.append(achievement)
                
            elif user_data['reduction_streak'] >= 2 and 'reduction_streak_2' not in [a['id'] for a in user_data['achievements']]:
                achievement = ACHIEVEMENTS['reduction_streak_2'].copy()
                achievement['id'] = 'reduction_streak_2'
                achievement['date'] = datetime.now().strftime('%Y-%m-%d')
                
                user_data['achievements'].append(achievement)
                user_data['total_points'] += achievement['points']
                new_achievements.append(achievement)
    
    # Update history
    user_data['history'].append({
        'date': datetime.now().strftime('%Y-%m-%d'),
        'kwh_used': current_usage,
        'points_earned': sum(a['points'] for a in new_achievements)
    })
    
    # Update last bill info
    user_data['last_bill_kwh'] = current_usage
    user_data['last_bill_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Save updated data
    save_user_gamification_data(user_data)
    
    return user_data, new_achievements

def get_energy_level(points):
    """
    Get the energy level based on points
    
    Args:
        points: The user's total points
        
    Returns:
        Dictionary with level information
    """
    levels = [
        {'name': 'Energy Novice', 'min_points': 0, 'max_points': 49},
        {'name': 'Energy Explorer', 'min_points': 50, 'max_points': 99},
        {'name': 'Energy Expert', 'min_points': 100, 'max_points': 199},
        {'name': 'Energy Master', 'min_points': 200, 'max_points': 349},
        {'name': 'Energy Guru', 'min_points': 350, 'max_points': 499},
        {'name': 'Energy Legend', 'min_points': 500, 'max_points': float('inf')}
    ]
    
    for level in levels:
        if level['min_points'] <= points <= level['max_points']:
            # Calculate progress to next level
            if level['max_points'] < float('inf'):
                total_range = level['max_points'] - level['min_points']
                progress = (points - level['min_points']) / total_range * 100
                next_level = levels[levels.index(level) + 1]['name']
            else:
                progress = 100
                next_level = None
            
            return {
                'name': level['name'],
                'points': points,
                'progress': round(progress, 1),
                'next_level': next_level,
                'points_to_next_level': level['max_points'] - points + 1 if level['max_points'] < float('inf') else 0
            }
    
    # Default fallback
    return {'name': 'Energy Novice', 'points': points, 'progress': 0, 'next_level': 'Energy Explorer', 'points_to_next_level': 50}

def generate_gamification_summary(user_data, new_achievements=None):
    """
    Generate a summary of the user's gamification status
    
    Args:
        user_data: User's gamification data
        new_achievements: List of newly earned achievements
        
    Returns:
        Dictionary with gamification summary
    """
    if new_achievements is None:
        new_achievements = []
    
    level_info = get_energy_level(user_data['total_points'])
    
    # Get recent achievements (up to 5)
    recent_achievements = sorted(
        user_data['achievements'], 
        key=lambda a: a.get('date', '2000-01-01'),
        reverse=True
    )[:5]
    
    # Get progress history
    usage_history = user_data['history']
    
    return {
        'total_points': user_data['total_points'],
        'level': level_info,
        'new_achievements': new_achievements,
        'recent_achievements': recent_achievements,
        'reduction_streak': user_data['reduction_streak'],
        'history': usage_history
    }