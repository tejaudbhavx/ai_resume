import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env
load_dotenv()

# MongoDB connection (ensure your .env contains MONGO_URI)
MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client['job_data']
# Assume resumes and job descriptions are stored in these collections:
resume_collection = db['resumes']
job_collection = db['job_descriptions']

# Create FastAPI instance
app = FastAPI()

def calculate_match_percentage(resume_text: str, jd_text: str) -> float:
    """
    Converts the resume and job description text into TF-IDF vectors and computes
    the cosine similarity, returning a match percentage.
    """
    try:
        # Initialize TF-IDF Vectorizer (with English stop words)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        match_percentage = float(similarity_matrix[0][0] * 100)
        return match_percentage
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing similarity: {e}")

@app.get("/match/")
async def match_resume_to_jd(resume_filename: str, jd_filename: str):
    """
    Matches a resume with a job description by retrieving both documents from MongoDB
    and computing a match percentage based on TF-IDF cosine similarity.
    
    Query Parameters:
      - resume_filename: The file name of the uploaded resume.
      - jd_filename: The file name of the uploaded job description.
    """
    # Retrieve the resume document from MongoDB
    resume_doc = resume_collection.find_one({"file_name": resume_filename})
    if not resume_doc:
        raise HTTPException(status_code=404, detail=f"Resume with filename '{resume_filename}' not found.")
    
    # Retrieve the job description document from MongoDB
    jd_doc = job_collection.find_one({"file_name": jd_filename})
    if not jd_doc:
        raise HTTPException(status_code=404, detail=f"Job description with filename '{jd_filename}' not found.")
    
    # Get text content from the documents
    resume_text = resume_doc.get("all_content", "")
    jd_text = jd_doc.get("content", "")
    
    if not resume_text or not jd_text:
        raise HTTPException(status_code=400, detail="One of the documents has no text content.")
    
    # Calculate match percentage using TF-IDF and cosine similarity
    match_percentage = calculate_match_percentage(resume_text, jd_text)
    
    # Return the result as a JSON response
    return JSONResponse(content={
        "resume_file": resume_filename,
        "job_description_file": jd_filename,
        "match_percentage": match_percentage
    })
