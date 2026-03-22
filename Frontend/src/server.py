from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import requests
import os

load_dotenv()

app = FastAPI()

# allow React to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SJSU_SCHOOL_ID = "U2Nob29sLTg4MQ=="

headers = {
    "Cookie": os.getenv("RMP_COOKIE") or "",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.ratemyprofessors.com/",
    "Origin": "https://www.ratemyprofessors.com/",
    "Content-Type": "application/json"
}

query = """
query NewSearchTeachersQuery(
  $query: TeacherSearchQuery!
  $count: Int
) {
  newSearch {
    teachers(query: $query, first: $count) {
      edges {
        node {
          legacyId
          firstName
          lastName
          department
          avgRating
          avgDifficulty
          numRatings
        }
      }
    }
  }
}
"""

@app.get("/search")
def search_professors(name: str):
    variables = {
        "query": {
            "text": name,
            "schoolID": SJSU_SCHOOL_ID,
            "fallback": True
        },
        "count": 10
    }

    try:
        response = requests.post(
            "https://www.ratemyprofessors.com/graphql",
            json={"query": query, "variables": variables},
            headers=headers
        )

        if response.status_code != 200:
            return {"error": "failed request"}

        data = response.json()

        edges = data["data"]["newSearch"]["teachers"]["edges"]

        professors = []

        for edge in edges:
            node = edge["node"]

            prof = {
                "id": node["legacyId"],
                "name": f"{node['firstName']} {node['lastName']}",
                "department": node["department"],
                "rating": node["avgRating"],
                "difficulty": node["avgDifficulty"],
                "numRatings": node["numRatings"]
            }

            professors.append(prof)

        return professors

    except Exception as e:
        return {"error": str(e)}