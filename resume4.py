import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cohere
from google import genai
from typing import List
import io
import fitz  # PyMuPDF
import docx
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import json
import datetime

# Load environment variables (for API keys)
from dotenv import load_dotenv
load_dotenv()

# API keys from .env file
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# MongoDB URI from your provided details
uri = "mongodb+srv://tejarachakonda:teja123@cluster0.9agko.mongodb.net/"

# Initialize Cohere client
co = cohere.Client(api_key=COHERE_API_KEY)

# Initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# MongoDB connection setup
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['user_data']  # Database name
collection = db['documents']  # Collection name

COHERE_EMBEDDING_MODEL = 'embed-english-v3.0'

# FastAPI app instance
app = FastAPI()

# Function to fetch embeddings using Cohere
def fetch_embeddings(texts: List[str], embedding_type: str = 'search_document') -> List[List[float]]:
    try:
        # Fetch embeddings for the provided texts
        results = co.embed(
            texts=texts,
            model=COHERE_EMBEDDING_MODEL,
            input_type=embedding_type
        ).embeddings
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohere embedding fetch failed: {e}")

# Function to synthesize answer using Gemini for extracting experience and skills
def synthesize_answer(question: str, context: List[str]) -> str:
    context_str = '\n'.join(context)
    prompt = f"""
    Extract ONLY the total years of experience and list of skills from the following document.

    ---------------------
    {context_str}
    ---------------------
    Provide the answer in the format:
    Years of Experience: <number> \n 
    Skills: <comma-separated list>
    """
    
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini API: {e}")

# Function to extract entire content, technical skills, and experience
@app.post("/extract-experience-skills/")
async def extract_experience_skills(file: UploadFile = File(...)):
    try:
        file_extension = file.filename.split('.')[-1].lower()
        file_content = await file.read()

        # Determine file type and extract text
        if file_extension == 'pdf':
            doc = fitz.open(stream=file_content, filetype='pdf')
            texts = [page.get_text() for page in doc]
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(file_content))
            texts = [para.text for para in doc.paragraphs]
        elif file_extension == 'txt':
            texts = file_content.decode('utf-8').splitlines()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Extract experience and skills using Gemini
        answer = synthesize_answer("Extract total years of experience and skills.", texts)

        # Extract all content as one large text
        all_content = "\n".join(texts)

        # Generate embeddings for the extracted text
        embeddings = fetch_embeddings([all_content])

        # Parse the response for experience and skills
        experience_match = None
        skills_match = None

        # Example basic string matching for the "Years of Experience" and "Skills"
        if "Years of Experience:" in answer:
            experience_match = answer.split("Years of Experience:")[1].split("\n")[0].strip()
        if "Skills:" in answer:
            skills_match = answer.split("Skills:")[1].strip()

        # Prepare the document data to store in MongoDB
        document_data = {
            "file_name": file.filename,
            "all_content": all_content,
            "technical_skills": skills_match if skills_match else "N/A",
            "years_of_experience": experience_match if experience_match else "N/A",
            "embeddings": json.dumps(embeddings),  # Store embeddings as JSON
            "uploaded_at": datetime.datetime.utcnow()  # Timestamp for the document upload
        }

        # Insert the document data into MongoDB collection
        result = collection.insert_one(document_data)

        # Return the response with the MongoDB insertion result
        return JSONResponse(content={"document_id": str(result.inserted_id), "answer": answer})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the document: {e}")
