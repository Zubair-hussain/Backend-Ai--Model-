from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from exploreit_ai import ExploreItAI

# Initialize FastAPI app
app = FastAPI()

# Initialize AI Recommendation class
ai = ExploreItAI()

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body schema
class UserInput(BaseModel):
    user_id: str
    age: int = None
    query: str = None

# 🟢 Endpoint: Onboard a new user
@app.post("/onboard")
def onboard_user(user: UserInput):
    return {"recommendations": ai.onboard_new_user(user.user_id, user.age or 25)}

# 🟡 Endpoint: Get AI recommendations
@app.post("/recommend")
def recommend(user: UserInput):
    return {"recommendations": ai.get_recommendations(user.user_id)}

# 🔵 Endpoint: Handle search query
@app.post("/search")
def search(user: UserInput):
    results = ai.process_search_query(user.user_id, user.query or "")
    return {
        "results": [
            {
                "event_id": r["event"]["event_id"],
                "name": r["event"]["name"],
                "category": r["event"]["category"],
                "score": r["relevance_score"]
            } for r in results
        ]
    }

# 🟣 Optional: Test route for {"hello": "Zubair"}
@app.post("/hello")
async def say_hello(data: dict):
    name = data.get("hello", "Guest")
    return {"message": f"Hello, {name}!"}


print("FastAPI backend loaded successfully!")

@app.post("/recommend")
def recommend(user: UserInput):
    print(f" Fetching recommendations for {user.user_id}")
    return {"recommendations": ai.get_recommendations(user.user_id)}

