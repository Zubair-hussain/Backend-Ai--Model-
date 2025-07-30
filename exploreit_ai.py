import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class ExploreItAI:
    def __init__(self):
        self.setup_database()
        self.initialize_models()
        self.user_profiles = {}
        self.interaction_weights = {
            'view': 1.0,
            'click': 2.0,
            'bookmark': 3.0,
            'share': 4.0,
            'attend': 5.0
        }

    def setup_database(self):
        self.conn = sqlite3.connect('exploreit.db', check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                age INTEGER,
                initial_interests TEXT,
                created_at TIMESTAMP,
                total_interactions INTEGER DEFAULT 0
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                event_id TEXT,
                interaction_type TEXT,
                duration INTEGER,
                timestamp TIMESTAMP,
                event_category TEXT,
                satisfaction_rating INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                search_query TEXT,
                results_clicked TEXT,
                timestamp TIMESTAMP,
                search_success BOOLEAN
            )
        ''')
        self.conn.commit()
        print("\u2705 Database initialized successfully")

    def initialize_models(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.satisfaction_model = XGBClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.events_data = self.generate_sample_events()
        self.train_initial_models()

    def generate_sample_events(self):
        categories = [
            "Adventure Sports", "Cultural Heritage", "Music Festivals", "Food & Dining",
            "Art Exhibitions", "Beach Activities", "Mountain Hiking", "Historical Tours",
            "Wildlife Safari", "Photography Workshops", "Spiritual Retreats", "Wine Tasting",
            "Water Sports", "City Walking Tours", "Night Markets", "Local Festivals"
        ]
        events = []
        for i in range(200):
            category = np.random.choice(categories)
            events.append({
                'event_id': f'event_{i:03d}',
                'name': f'{category} Experience {i}',
                'category': category,
                'description': f'Amazing {category.lower()} experience with local guides',
                'location': f'Location {i%20}',
                'price': np.random.randint(20, 200),
                'rating': np.random.uniform(3.5, 5.0),
                'popularity_score': np.random.uniform(0.1, 1.0)
            })
        return pd.DataFrame(events)

    def train_initial_models(self):
        sample_data = []
        for _ in range(1000):
            age = np.random.randint(18, 70)
            acc = np.random.uniform(60, 95)
            engage = np.random.uniform(0.1, 1.0)
            score = 3 + (acc - 60) / 20 + engage
            sample_data.append([age, acc, engage, int(min(5, max(3, score)))])
        df = pd.DataFrame(sample_data, columns=['age', 'rec_accuracy', 'engagement', 'satisfaction'])
        X = df[['age', 'rec_accuracy', 'engagement']]
        y = df['satisfaction'] - 3
        self.satisfaction_model.fit(X, y)
        print("\u2705 Initial models trained")

    def predict_satisfaction(self, age, acc, engage):
        X = np.array([[age, acc, engage]])
        return self.satisfaction_model.predict(X)[0] + 3  # Adjust back to 1-5 scale

    def onboard_new_user(self, user_id, age, preferences=None):
        if preferences is None:
            preferences = self.interactive_preference_collection()
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO user_profiles (user_id, age, initial_interests, created_at)
            VALUES (?, ?, ?, ?)
        ''', (user_id, age, json.dumps(preferences), datetime.now()))
        self.conn.commit()
        self.user_profiles[user_id] = {
            'age': age,
            'interests': preferences,
            'interaction_history': [],
            'search_patterns': {},
            'preference_weights': {pref: 1.0 for pref in preferences},
            'last_updated': datetime.now()
        }
        print(f"\u2705 User {user_id} onboarded with preferences: {preferences}")
        return self.get_recommendations(user_id, bootstrap=True)

    def interactive_preference_collection(self):
        categories = [
            "Adventure Sports", "Cultural Heritage", "Music Festivals", "Food & Dining",
            "Art Exhibitions", "Beach Activities", "Mountain Hiking", "Historical Tours",
            "Wildlife Safari", "Photography Workshops", "Spiritual Retreats", "Wine Tasting"
        ]
        selected = list(np.random.choice(categories, np.random.randint(3, 6), replace=False))
        print(f"\U0001F3A8 Selected preferences: {selected}")
        return selected

    def track_user_interaction(self, user_id, event_id, interaction_type, duration=None, satisfaction=None):
        if user_id not in self.user_profiles:
            print(f"\u274C Error: User {user_id} not found")
            return
        event = self.events_data[self.events_data['event_id'] == event_id]
        if event.empty:
            print(f"\u274C Error: Event {event_id} not found")
            return
        event_category = event.iloc[0]['category']
        # Predict satisfaction if not provided
        if satisfaction is None and duration is not None:
            profile = self.user_profiles[user_id]
            acc = self.calculate_recommendation_accuracy(user_id, event_category)
            engage = min(duration / 300, 1.0)  # Normalize engagement
            satisfaction = self.predict_satisfaction(profile['age'], acc, engage)
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO user_interactions 
            (user_id, event_id, interaction_type, duration, timestamp, event_category, satisfaction_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, event_id, interaction_type, duration, datetime.now(), event_category, satisfaction))
        self.conn.commit()
        self.update_user_preferences(user_id, event_category, interaction_type, duration, satisfaction)
        print(f"\U0001F4CA Tracked interaction: {user_id} → {interaction_type} on {event_id} (satisfaction: {satisfaction})")

    def calculate_recommendation_accuracy(self, user_id, event_category):
        profile = self.user_profiles[user_id]
        history = profile.get('interaction_history', [])
        relevant = [i for i in history if i['category'] == event_category]
        if not relevant:
            return 60.0  # Default accuracy
        avg_satisfaction = np.mean([i.get('satisfaction', 3) for i in relevant if i.get('satisfaction')])
        return 60.0 + (avg_satisfaction - 3) * 10  # Scale to 60-100

    def update_user_preferences(self, user_id, event_category, interaction_type, duration, satisfaction):
        user_profile = self.user_profiles[user_id]
        base_score = self.interaction_weights.get(interaction_type, 1.0)
        duration_bonus = min(duration / 300, 2.0) if duration else 1.0
        satisfaction_bonus = (satisfaction / 5.0) if satisfaction else 1.0
        interaction_score = base_score * duration_bonus * satisfaction_bonus
        if event_category in user_profile['preference_weights']:
            user_profile['preference_weights'][event_category] += interaction_score * 0.1
        else:
            user_profile['preference_weights'][event_category] = interaction_score * 0.1
        user_profile['interaction_history'].append({
            'category': event_category,
            'type': interaction_type,
            'score': interaction_score,
            'satisfaction': satisfaction,
            'timestamp': datetime.now()
        })
        user_profile['interaction_history'] = user_profile['interaction_history'][-100:]
        user_profile['last_updated'] = datetime.now()
        self.apply_preference_decay(user_id)

    def apply_preference_decay(self, user_id):
        user_profile = self.user_profiles[user_id]
        for category in user_profile['preference_weights']:
            user_profile['preference_weights'][category] *= 0.95
            user_profile['preference_weights'][category] = max(0.1, user_profile['preference_weights'][category])

    def process_search_query(self, user_id, query, clicked_results=None):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO search_history (user_id, search_query, results_clicked, timestamp, search_success)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, query, json.dumps(clicked_results) if clicked_results else None, datetime.now(), clicked_results is not None))
        self.conn.commit()
        search_intent = self.extract_search_intent(query)
        if user_id in self.user_profiles:
            self.update_search_patterns(user_id, query, search_intent, clicked_results)
        results = self.generate_search_results(query, user_id)
        print(f"\U0001F50D Processed search for {user_id}: '{query}' → {len(results)} results")
        return results

    def extract_search_intent(self, query):
        query_lower = query.lower()
        category_keywords = {
            'Adventure Sports': ['adventure', 'sports', 'extreme', 'thrill', 'climbing'],
            'Cultural Heritage': ['culture', 'heritage', 'traditional', 'history', 'museum'],
            'Music Festivals': ['music', 'festival', 'concert', 'band', 'live'],
            'Food & Dining': ['food', 'dining', 'restaurant', 'cuisine', 'cooking'],
            'Art Exhibitions': ['art', 'exhibition', 'gallery', 'painting', 'sculpture'],
            'Beach Activities': ['beach', 'ocean', 'sea', 'swimming', 'surfing'],
            'Mountain Hiking': ['mountain', 'hiking', 'trekking', 'trail', 'peak'],
            'Historical Tours': ['historical', 'tour', 'ancient', 'monument', 'castle']
        }
        return [cat for cat, keys in category_keywords.items() if any(k in query_lower for k in keys)]

    def update_search_patterns(self, user_id, query, intent, clicked_results):
        user_profile = self.user_profiles[user_id]
        for keyword in query.lower().split():
            user_profile['search_patterns'][keyword] = user_profile['search_patterns'].get(keyword, 0) + 1
        if clicked_results:
            for category in intent:
                user_profile['preference_weights'][category] = user_profile['preference_weights'].get(category, 0) + 0.2

    def generate_search_results(self, query, user_id):
        user_prefs = self.user_profiles.get(user_id, {}).get('preference_weights', {})
        query_lower = query.lower()
        event_texts = self.events_data['name'] + ' ' + self.events_data['category'] + ' ' + self.events_data['description']
        tfidf_matrix = self.vectorizer.fit_transform(event_texts)
        query_vector = self.vectorizer.transform([query_lower])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        matching = []
        for idx, event in self.events_data.iterrows():
            score = similarities[idx] * 3
            if query_lower in event['name'].lower(): score += 3
            if query_lower in event['category'].lower(): score += 2
            if query_lower in event['description'].lower(): score += 1
            if event['category'] in user_prefs:
                score *= (1 + user_prefs[event['category']])
            if score > 0:
                matching.append({'event': event, 'relevance_score': score})
        return sorted(matching, key=lambda x: x['relevance_score'], reverse=True)[:10]

    def get_recommendations(self, user_id, num_recommendations=5, bootstrap=False):
        if user_id not in self.user_profiles:
            print(f"\u274C Error: User {user_id} not found")
            return []
        profile = self.user_profiles[user_id]
        weights = profile['preference_weights']
        event_scores = []
        for _, event in self.events_data.iterrows():
            score = self.calculate_event_score(event, weights, profile, bootstrap)
            event_scores.append({'event': event, 'score': score})
        event_scores.sort(key=lambda x: x['score'], reverse=True)
        recommendations = [
            {
                'event_id': item['event']['event_id'],
                'name': item['event']['name'],
                'category': item['event']['category'],
                'score': item['score'],
                'reason': self.generate_recommendation_reason(item['event'], weights)
            }
            for item in event_scores[:num_recommendations]
        ]
        print(f"\U0001F3AF Generated {len(recommendations)} recommendations for {user_id}")
        return recommendations

    def calculate_event_score(self, event, weights, profile, bootstrap):
        base = event['popularity_score'] * 0.3 + (event['rating'] / 5.0) * 0.2
        pref_score = weights.get(event['category'], 0.1) * 0.4
        inter_bonus = self.calculate_interaction_bonus(event, profile) if not bootstrap else 0
        diversity = self.calculate_diversity_factor(event, profile)
        return base + pref_score + inter_bonus + diversity

    def calculate_interaction_bonus(self, event, profile):
        history = profile.get('interaction_history', [])
        similar = [i for i in history if i['category'] == event['category']]
        if similar:
            return min(np.mean([i['score'] for i in similar]) * 0.1, 0.3)
        return 0

    def calculate_diversity_factor(self, event, profile):
        history = profile.get('interaction_history', [])[-20:]
        same = [i for i in history if i['category'] == event['category']]
        if len(same) > 5:
            return -0.1
        elif not same:
            return 0.1
        return 0

    def generate_recommendation_reason(self, event, weights):
        cat = event['category']
        if weights.get(cat, 0) > 1.0:
            return f"Based on your interest in {cat}"
        if event['rating'] > 4.5:
            return f"Highly rated {cat} experience"
        if event['popularity_score'] > 0.8:
            return f"Popular {cat} activity"
        return f"Recommended {cat} experience"

    def get_user_analytics(self, user_id):
        if user_id not in self.user_profiles:
            print(f"\u274C Error: User {user_id} not found")
            return None
        analytics = {}
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM user_interactions WHERE user_id=?', (user_id,))
        analytics["total_interactions"] = cursor.fetchone()[0]
        cursor.execute('''
            SELECT event_category, COUNT(*) as freq 
            FROM user_interactions 
            WHERE user_id=? 
            GROUP BY event_category 
            ORDER BY freq DESC 
            LIMIT 3
        ''', (user_id,))
        analytics["top_categories"] = [row[0] for row in cursor.fetchall()]
        cursor.execute('''
            SELECT interaction_type, COUNT(*), AVG(duration) 
            FROM user_interactions 
            WHERE user_id=? 
            GROUP BY interaction_type
        ''', (user_id,))
        analytics["interaction_types"] = {
            row[0]: {'count': row[1], 'avg_duration': round(row[2] or 0, 2)}
            for row in cursor.fetchall()
        }
        cursor.execute('''
            SELECT COUNT(*), SUM(CASE WHEN search_success THEN 1 ELSE 0 END) 
            FROM search_history WHERE user_id=?
        ''', (user_id,))
        total_searches, successful = cursor.fetchone()
        analytics["search_stats"] = {
            "total_searches": total_searches,
            "successful_searches": successful or 0,
            "success_rate": round((successful / total_searches * 100), 2) if total_searches else 0
        }
        cursor.execute('''
            SELECT search_query 
            FROM search_history 
            WHERE user_id=? 
            ORDER BY timestamp DESC 
            LIMIT 5
        ''', (user_id,))
        analytics["recent_searches"] = [row[0] for row in cursor.fetchall()]
        profile = self.user_profiles[user_id]
        analytics["profile_age_days"] = (datetime.now() - profile['last_updated']).days
        print(f"\U0001F4CA Analytics generated for {user_id}")
        return analytics

def test_enhanced_system():
    print("\n\U0001F9EA Running test_enhanced_system()...\n")
    ai = ExploreItAI()
    user_id = "user_001"
    print("\U0001F464 Onboarding user...")
    ai.onboard_new_user(user_id, age=28)
    print("\U0001F5B1 Simulating interactions...")
    ai.track_user_interaction(user_id, "event_005", "click", duration=45, satisfaction=4)
    ai.track_user_interaction(user_id, "event_007", "view", duration=20)
    ai.track_user_interaction(user_id, "event_012", "bookmark", duration=30)
    print("\U0001F50D Processing search query...")
    ai.process_search_query(user_id, "mountain hiking", ["event_020", "event_021"])
    print("\U0001F3AF Getting recommendations...")
    recommendations = ai.get_recommendations(user_id)
    for i, r in enumerate(recommendations, 1):
        print(f"{i}. {r['name']} ({r['category']}) → {r['reason']}")
    print("\n\U0001F4CA Fetching user analytics...")
    analytics = ai.get_user_analytics(user_id)
    print(json.dumps(analytics, indent=4))
    print("\n\U00002705 Test completed.\n")