import os
import datetime
from io import BytesIO
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from docx import Document
import fitz  # PyMuPDF for PDF

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client['job_data']
job_collection = db['job_descriptions']

# Initialize FastAPI router
jd_router = APIRouter()

def extract_text_from_docx(file_content: bytes) -> str:
    """
    Extracts text from a DOCX file using python-docx.
    """
    try:
        doc = Document(BytesIO(file_content))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing DOCX file: {e}")

def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF file: {e}")

@jd_router.post("/upload/")
async def upload_job_description(file: UploadFile = File(...)):
    """
    Handles job description uploads and stores data in MongoDB.
    Supports DOCX, PDF, and TXT file formats.
    """
    try:
        file_extension = file.filename.split('.')[-1].lower()
        file_content = await file.read()

        # Extract text based on file type
        if file_extension == "docx":
            jd_text = extract_text_from_docx(file_content)
        elif file_extension == "pdf":
            jd_text = extract_text_from_pdf(file_content)
        elif file_extension == "txt":
            try:
                jd_text = file_content.decode('utf-8')
            except UnicodeDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Error decoding txt file: {e}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Prepare the document data for MongoDB
        document_data = {
            "file_name": file.filename,
            "content": jd_text,
            "uploaded_at": datetime.datetime.utcnow()
        }

        # Insert the document into MongoDB
        job_collection.insert_one(document_data)
        return JSONResponse(content={"message": "Job description uploaded successfully", "file_name": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the job description: {e}")

