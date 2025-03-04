# import os
# import io
# import json
# import datetime
# import fitz  # PyMuPDF for PDF
# import docx
# import boto3
# from dotenv import load_dotenv
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from pymongo import MongoClient
# from pymongo.server_api import ServerApi # type: ignore
# import cohere
# from google import genai
# from typing import List

# # Load environment variables
# load_dotenv()

# # API Keys from .env file
# COHERE_API_KEY = os.getenv('COHERE_API_KEY')
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
# AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
# S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

# # Initialize external services
# co = cohere.Client(api_key=COHERE_API_KEY)
# gemini_client = genai.Client(api_key=GEMINI_API_KEY)
# s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# # MongoDB connection
# MONGO_URI = "mongodb+srv://tejarachakonda:teja123@cluster0.9agko.mongodb.net/"
# client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
# db = client['user_data']
# collection = db['documents']

# COHERE_EMBEDDING_MODEL = 'embed-english-v3.0'

# # FastAPI instance
# app = FastAPI()

# # ------------------ S3 Functions ------------------

# def upload_to_s3(file_content: bytes, file_name: str) -> str:
#     """
#     Uploads a file to Amazon S3 and returns the S3 URL.
#     """
#     try:
#         s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=file_name, Body=file_content)
#         return f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{file_name}"
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {e}")

# def get_resume_from_s3(file_name: str) -> bytes:
#     """
#     Retrieves a file from Amazon S3.
#     """
#     try:
#         response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_name)
#         return response['Body'].read()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to retrieve file from S3: {e}")

# # ------------------ Cohere & Gemini Functions ------------------

# def fetch_embeddings(texts: List[str], embedding_type: str = 'search_document') -> List[List[float]]:
#     """
#     Fetches text embeddings using Cohere API.
#     """
#     try:
#         results = co.embed(
#             texts=texts,
#             model=COHERE_EMBEDDING_MODEL,
#             input_type=embedding_type
#         ).embeddings
#         return results
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Cohere embedding fetch failed: {e}")

# def synthesize_answer(question: str, context: List[str]) -> str:
#     """
#     Uses Gemini AI to extract experience & skills from resume text.
#     """
#     context_str = '\n'.join(context)
#     prompt = f"""
#     Extract ONLY the total years of experience and list of skills from the following document.

#     ---------------------
#     {context_str}
#     ---------------------
#     Provide the answer in the format:
#     Years of Experience: <number> \n 
#     Skills: <comma-separated list>
#     """

#     try:
#         response = gemini_client.models.generate_content(
#             model="gemini-2.0-flash",
#             contents=[prompt]
#         )
#         return response.text
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error communicating with Gemini API: {e}")

# # ------------------ API Endpoints ------------------

# @app.post("/extract-experience-skills/")
# async def extract_experience_skills(file: UploadFile = File(...)):
#     """
#     Handles resume uploads, extracts experience & skills, and stores data in MongoDB.
#     """
#     try:
#         file_extension = file.filename.split('.')[-1].lower()
#         file_content = await file.read()
#         s3_file_path = f"resumes/{file.filename}"

#         # Check if the resume is already uploaded
#         existing_doc = collection.find_one({"file_name": file.filename})
#         if existing_doc:
#             # Fetch resume from S3 instead of re-uploading
#             file_content = get_resume_from_s3(s3_file_path)
#             print("🔄 Using existing resume from S3")

#         else:
#             # Upload to S3 and save path in MongoDB
#             s3_url = upload_to_s3(file_content, s3_file_path)
#             print("✅ New resume uploaded to S3")

#         # Extract text from resume
#         texts = []
#         if file_extension == 'pdf':
#             doc = fitz.open(stream=file_content, filetype='pdf')
#             texts = [page.get_text() for page in doc]
#         elif file_extension == 'docx':
#             doc = docx.Document(io.BytesIO(file_content))
#             texts = [para.text for para in doc.paragraphs]
#         elif file_extension == 'txt':
#             texts = file_content.decode('utf-8').splitlines()
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file type")

#         # Extract experience & skills using Gemini AI
#         answer = synthesize_answer("Extract total years of experience and skills.", texts)

#         # Process extracted data
#         experience_match = None
#         skills_match = None
#         if "Years of Experience:" in answer:
#             experience_match = answer.split("Years of Experience:")[1].split("\n")[0].strip()
#         if "Skills:" in answer:
#             skills_match = answer.split("Skills:")[1].strip()

#         # Generate embeddings for text content
#         all_content = "\n".join(texts)
#         embeddings = fetch_embeddings([all_content])

#         # Prepare document data for MongoDB
#         document_data = {
#             "file_name": file.filename,
#             "s3_url": f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_file_path}",
#             "all_content": all_content,
#             "technical_skills": skills_match if skills_match else "N/A",
#             "years_of_experience": experience_match if experience_match else "N/A",
#             "embeddings": json.dumps(embeddings),
#             "uploaded_at": datetime.datetime.utcnow()
#         }

#         # Insert or update document in MongoDB
#         if existing_doc:
#             collection.update_one({"file_name": file.filename}, {"$set": document_data})
#             print("📝 Updated existing record in MongoDB")
#         else:
#             collection.insert_one(document_data)
#             print("📌 New record inserted in MongoDB")

#         return JSONResponse(content={"document_id": file.filename, "answer": answer})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing the document: {e}")
import os
import io
import json
import datetime
import fitz  # PyMuPDF for PDF
import docx
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import cohere
from google import genai
from typing import List
# Load environment variables
load_dotenv()
# API Keys from .env file
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
# Initialize external services
co = cohere.Client(api_key=COHERE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
# MongoDB connection
#MONGO_URI = "mongodb+srv://krishnasai:krishna132@cluster0.wjww1.mongodb.net/"
MONGO_URI="mongodb+srv://tejarachakonda:teja123@cluster0.9agko.mongodb.net/"
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client['user_data']
collection = db['documents']
COHERE_EMBEDDING_MODEL = 'embed-english-v3.0'
# FastAPI instance
app = FastAPI()
# ------------------ S3 Functions ------------------
def upload_to_s3(file_content: bytes, file_name: str) -> str:
    """
    Uploads a file to Amazon S3 and returns the S3 URL.
    """
    try:
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=file_name, Body=file_content)
        return f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{file_name}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {e}")
def get_resume_from_s3(file_name: str) -> bytes:
    """
    Retrieves a file from Amazon S3.
    """
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_name)
        return response['Body'].read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file from S3: {e}")
# ------------------ Cohere & Gemini Functions ------------------
def fetch_embeddings(texts: List[str], embedding_type: str = 'search_document') -> List[List[float]]:
    """
    Fetches text embeddings using Cohere API.
    """
    try:
        results = co.embed(
            texts=texts,
            model=COHERE_EMBEDDING_MODEL,
            input_type=embedding_type
        ).embeddings
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohere embedding fetch failed: {e}")
def synthesize_answer(question: str, context: List[str]) -> str:
    """
    Uses Gemini AI to extract experience & skills from resume text.
    """
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
# ------------------ API Endpoints ------------------
@app.post("/extract-experience-skills/")
async def extract_experience_skills(file: UploadFile = File(...)):
    """
    Handles resume uploads, extracts experience & skills, and stores data in MongoDB.
    """
    try:
        file_extension = file.filename.split('.')[-1].lower()
        file_content = await file.read()
        # Debug: Print file extension and content length
        print(f"Received file with extension: {file_extension}")
        print(f"File content length: {len(file_content)} bytes")
        s3_file_path = f"resumes/{file.filename}"
        # Check if the resume is already uploaded
        existing_doc = collection.find_one({"file_name": file.filename})
        if existing_doc:
            # Fetch resume from S3 instead of re-uploading
            file_content = get_resume_from_s3(s3_file_path)
            print(":arrows_counterclockwise: Using existing resume from S3")
        else:
            # Upload to S3 and save path in MongoDB
            s3_url = upload_to_s3(file_content, s3_file_path)
            print(":white_check_mark: New resume uploaded to S3")
        # Extract text from resume
        texts = []
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
        # Debug: Print extracted text
        print(f"Extracted {len(texts)} pages/paragraphs from the document.")
        # Extract experience & skills using Gemini AI
        answer = synthesize_answer("Extract total years of experience and skills.", texts)
        # Debug: Print the answer from Gemini
        print(f"Gemini response: {answer}")
        # Process extracted data
        experience_match = None
        skills_match = None
        if "Years of Experience:" in answer:
            experience_match = answer.split("Years of Experience:")[1].split("\n")[0].strip()
        if "Skills:" in answer:
            skills_match = answer.split("Skills:")[1].strip()
        # Debug: Print experience and skills
        print(f"Experience: {experience_match}")
        print(f"Skills: {skills_match}")
        # Generate embeddings for text content
        all_content = "\n".join(texts)
        embeddings = fetch_embeddings([all_content])
        # Prepare document data for MongoDB
        document_data = {
            "file_name": file.filename,
            "s3_url": f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_file_path}",
            "all_content": all_content,
            "technical_skills": skills_match if skills_match else "N/A",
            "years_of_experience": experience_match if experience_match else "N/A",
            "embeddings": json.dumps(embeddings),
            "uploaded_at": datetime.datetime.utcnow()
        }
        # Insert or update document in MongoDB
        if existing_doc:
            collection.update_one({"file_name": file.filename}, {"$set": document_data})
            print(":memo: Updated existing record in MongoDB")
        else:
            collection.insert_one(document_data)
            print(":pushpin: New record inserted in MongoDB")
        return JSONResponse(content={"document_id": file.filename, "answer": answer})
    except Exception as e:
        # Log the full exception for debugging
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing the document: {e}")
    
from fastapi import APIRouter

resume_router = APIRouter()

@resume_router.post("/upload/")
async def upload_resume(file: UploadFile = File(...)):
    # Your code to handle resume upload
    pass
