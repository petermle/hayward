import requests
import supabase
from datetime import datetime, timedelta, timezone

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.ratemyprofessors.com/",
    "Origin": "https://www.ratemyprofessors.com",
    "Content-Type": "application/json"
}

QUERY = """
query NewSearchTeachersQuery(
  $query: TeacherSearchQuery!
  $count: Int
) {
  newSearch {
    teachers(query: $query, first: $count) {
      didFallback
      edges {
        node {
          id
          legacyId
          firstName
          lastName
          department
          avgRating
          avgDifficulty
          numRatings
          wouldTakeAgainPercentRounded
        }
      }
    }
  }
}
"""


def get_school_id(school_name: str) -> str:
    query = """
    query NewSearchSchoolsQuery(
      $query: SchoolSearchQuery
    ) {
      newSearch {
        schools(query: $query) {
          edges {
            node {
              id
              legacyId
              name
              city
              state
            }
          }
        }
      }
    }
    """

    variables = {
        "query": {
            "text": school_name
        }
    }

    response = requests.post(
        "https://www.ratemyprofessors.com/graphql",
        json={"query": query, "variables": variables},
        headers=HEADERS
    )

    print(response.status_code)
    print(response.json())

    if response.status_code != 200:
        return None

    edges = response.json()["data"]["newSearch"]["schools"]["edges"]
    print(edges)
    if not edges:
        return None

    return edges[0]["node"]["id"]


def get_professors(name: str, school_id: str, count: int = 10) -> list:
    """Fetch professors from RMP by name and school"""
    variables = {
        "query": {
            "text": name,
            "schoolID": school_id,
            "fallback": True
        },
        "count": count
    }

    response = requests.post(
        "https://www.ratemyprofessors.com/graphql",
        json={"query": QUERY, "variables": variables},
        headers=HEADERS
    )

    if response.status_code != 200:
        return []

    edges = response.json()["data"]["newSearch"]["teachers"]["edges"]

    professors = []
    for edge in edges:
        node = edge["node"]
        professors.append({
            "legacy_id": node["legacyId"],
            "first_name": node["firstName"],
            "last_name": node["lastName"],
            "department": node["department"],
            "rating": node["avgRating"],
            "difficulty": node["avgDifficulty"],
            "num_ratings": node["numRatings"],
            "would_take_again": node["wouldTakeAgainPercentRounded"],
            "url": f"https://www.ratemyprofessors.com/professor/{node['legacyId']}"
        })

    return professors


def get_professors_cached(name: str, school_id: str) -> list:
    """Check cache first, only hit RMP if data is missing or stale"""
    cached = supabase.table("professors")\
        .select("*")\
        .ilike("last_name", f"%{name}%")\
        .eq("school_id", school_id)\
        .execute()
    if cached.data:
        last_updated = datetime.fromisoformat(cached.data[0]["last_updated"])
        if datetime.now(timezone.utc) - last_updated < timedelta(days=7):
            return cached.data

    # fetch fresh from RMP
    professors = get_professors(name, school_id)

    # upsert into cache
    for prof in professors:
        supabase.table("professors").upsert({
            "legacy_id": prof["legacy_id"],
            "first_name": prof["first_name"],
            "last_name": prof["last_name"],
            "department": prof["department"],
            "school_id": school_id,
            "avg_rating": prof["rating"],
            "avg_difficulty": prof["difficulty"],
            "num_ratings": prof["num_ratings"],
            "would_take_again": prof["would_take_again"],
            "rmp_url": prof["url"],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }, on_conflict="legacy_id").execute()

    return professors
