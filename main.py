from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import supabase
from backend.rmp import get_school_id, get_professors_cached

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/professors")
def search_professors(name: str, school: str):
    school_id = get_school_id(school)
    professors = get_professors_cached(name, school_id)
    return {"professors": professors}

@app.get("/recommendations/{professor_name}")
def get_recommendation(professor_name: str, course_code: str = None):
    data = supabase.table("recommendations")\
        .select("*")\
        .eq("professor_name", professor_name)\
        .execute()
    return {"recommendations": data.data}