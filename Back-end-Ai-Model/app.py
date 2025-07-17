import streamlit as st
from exploreit_ai import ExploreItAI

st.set_page_config(page_title="Explore-It AI Demo")
ai = ExploreItAI()

st.title("🎯 Explore-It AI Recommendation Demo")

# Onboard new user
st.subheader("🧠 Onboard New User")
user_id = st.text_input("User ID", "user_001")
age = st.slider("User Age", 15, 70, 25)

if st.button("Create Profile"):
    recs = ai.onboard_new_user(user_id, age)
    st.success("User onboarded successfully!")
    for r in recs:
        st.write(f"- {r['name']} ({r['category']}) — {r['reason']}")

# Interaction
st.subheader("📊 Simulate Interaction")
event_id = st.text_input("Event ID (e.g. event_005)")
interaction_type = st.selectbox("Interaction Type", ["view", "click", "bookmark", "share", "attend"])
duration = st.slider("Duration (seconds)", 0, 300, 30)
satisfaction = st.slider("Satisfaction (1-5)", 1, 5, 4)

if st.button("Submit Interaction"):
    ai.track_user_interaction(user_id, event_id, interaction_type, duration, satisfaction)
    st.success("Interaction Recorded")

# Get Recommendations
st.subheader("🎯 Get Personalized Recommendations")
if st.button("Get Recommendations"):
    recs = ai.get_recommendations(user_id, num_recommendations=5)
    for r in recs:
        st.write(f"- {r['name']} ({r['category']}) — Score: {r['score']:.2f}")

# Search
st.subheader("🔍 Search")
query = st.text_input("Search Query")

if st.button("Search"):
    results = ai.process_search_query(user_id, query)
    for item in results:
        ev = item['event']
        st.write(f"- {ev['name']} ({ev['category']}) — Relevance Score: {item['relevance_score']:.2f}")