# ExploreItAI: Intelligent Event Recommendation System

ExploreItAI is an intelligent recommendation system designed to enhance user discovery and engagement through personalized event suggestions. Built using Python, Scikit-learn, XGBoost, and SQLite, it models user behavior, search patterns, and interaction history to offer smart recommendations tailored to individual preferences.

##  Purpose

This module powers the **"Explore"** section of an application where users discover personalized local and interest-based experiences. Whether it’s music festivals, food & dining, mountain hiking, or art exhibitions — ExploreItAI adapts to user preferences dynamically to present the most relevant activities.

---

##  Features

###  User Onboarding & Preference Modeling
- Collects age and initial interest categories on sign-up.
- Dynamically adjusts preferences based on interactions (views, clicks, bookmarks, shares, and attendance).

###  Intelligent Search Processing
- Extracts search intent from free-text queries.
- Logs and learns from successful/failed search interactions to improve future relevance.

###  Personalized Event Recommendations
- Combines popularity, event rating, and user interest scores.
- Includes diversity boosting and preference decay to avoid repetitive suggestions.

###  Analytics & Insights
- Tracks user engagement metrics like interaction types, top categories, and satisfaction scores.
- Calculates recommendation accuracy and user profile freshness.

###  Persistent Storage (SQLite3)
- Stores user profiles, interactions, and search history in a local SQLite database.

###  Built-in Simulation for Testing
- `test_enhanced_system()` function simulates onboarding, interaction, searching, and provides recommendation output and analytics.

---

##  Example Use Case (Explore App Section)

In the **Explore** section of the app:
- A user is onboarded with their interests and age.
- As they browse and interact with events, their preferences are refined.
- The app displays personalized event cards like:
  > " Art Exhibition in Location 5 - Based on your interest in Art Exhibitions"

- Users can also search using natural language queries like:
  > *“Extreme sports in the mountains”*

- Recommendations are updated in real-time based on:
  - Search success rate
  - Interaction depth
  - Diversity of experiences
  - Satisfaction feedback

---

##  Technologies Used

- `pandas`, `numpy` — Data handling
- `sqlite3` — Lightweight database
- `scikit-learn` — Text vectorization (TF-IDF)
- `xgboost` — Satisfaction prediction
- `cosine_similarity` — Search result ranking

---

##  File Structure

- `ExploreItAI` — Main class encapsulating all core logic.
- `exploreit.db` — Local SQLite3 database (auto-created).
- `test_enhanced_system()` — Demo and testing flow.

---

##  How to Run

```bash
pip install pandas numpy scikit-learn xgboost
python your_script.py
python -m uvicorn main:app --reload
Make sure the script includes:


if __name__ == "__main__":
    test_enhanced_system()
 Final Note
ExploreItAI transforms static content discovery into a dynamic, personalized journey. It powers the Explore section with smart recommendations, adaptive preferences, and insightful analytics — enabling users to find the most meaningful experiences effortlessly.

