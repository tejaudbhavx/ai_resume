from fastapi import FastAPI
from s3 import resume_router  # Your resume code
from jd import jd_router       # Job description code

app = FastAPI()

app.include_router(resume_router, prefix="/resume")
app.include_router(jd_router, prefix="/job-description")
