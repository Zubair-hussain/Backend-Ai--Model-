
# ExploreIt AI Recommendation System

## Overview
ExploreIt AI is a complete recommendation system with behavioral learning capabilities. It provides personalized recommendations for events and activities based on user interactions, preferences, and search history.

## Features
- User Onboarding: Spotify-like interactive preference collection for new users
- Behavioral Learning: Tracks user interactions (view, click, bookmark, share, attend) to update preferences
- Search Integration: Processes search queries with intent analysis and learning
- Personalized Recommendations: Generates tailored recommendations using user profiles and interaction history
- Analytics: Provides comprehensive user behavior analytics
- Database Management: Uses SQLite for storing user profiles, interactions, and search history
- Machine Learning: Implements TF-IDF for text processing and XGBoost for satisfaction prediction

## Requirements
- python>=3.8
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- sqlite3

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd exploreit-ai


2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Import and initialize the system:
```python
from exploreit_ai import ExploreItAI
ai_system = ExploreItAI()
```

### Onboard a new user:
```python
user_id = "user_001"
initial_recommendations = ai_system.onboard_new_user(user_id, age=28)
```

### Track user interactions:
```python
ai_system.track_user_interaction(user_id, "event_005", "click", duration=45, satisfaction=4)
```

### Process search queries:
```python
search_results = ai_system.process_search_query(user_id, "mountain hiking adventure", ["event_020"])
```

### Get personalized recommendations:
```python
recommendations = ai_system.get_recommendations(user_id, num_recommendations=5)
```

### Access user analytics:
```python
analytics = ai_system.get_user_analytics(user_id)
```

## File Structure
```
exploreit-ai/
├── exploreit_ai.py     # Main system implementation
├── README.md          # This file
└── requirements.txt   # Dependencies
```

## Testing
Run the built-in test suite to verify system functionality:
```python
from exploreit_ai import test_enhanced_system
test_enhanced_system()
```

## Database Schema
- **user_profiles**: Stores user information (user_id, age, initial_interests, created_at, total_interactions)
- **user_interactions**: Tracks user interactions (user_id, event_id, interaction_type, duration, timestamp, event_category, satisfaction_rating)
- **search_history**: Records search queries (user_id, search_query, results_clicked, timestamp, search_success)

## Key Components
- **ExploreItAI Class**: Main class handling initialization, user management, and recommendations
- **ML Models**: Uses TF-IDF vectorizer for text processing and XGBoost for satisfaction prediction
- **Behavioral Learning**: Implements interaction weights and preference decay
- **Recommendation Engine**: Combines popularity, ratings, user preferences, and interaction history

## Future Improvements
- Add real-time model retraining
- Implement more sophisticated NLP for search intent
- Add content-based filtering
- Enhance diversity in recommendations
- Add support for collaborative filtering

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License
MIT License - Created by Zubair Hussain


