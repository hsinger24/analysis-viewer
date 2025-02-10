from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
import pandas as pd
import numpy as np
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
import ast
import sys
from io import StringIO
import contextlib
import traceback
from typing import Any, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv
import glob
import matplotlib
import logging
import asyncio

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def clean_for_json(obj):
    """Clean numeric values to make them JSON serializable"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        # Handle NaN and Infinity
        if pd.isna(obj) or np.isinf(obj):
            return None
        # Handle very large/small floats
        try:
            float_val = float(obj)
            if abs(float_val) > 1e308:  # Max JSON float
                return str(float_val)
            return float_val
        except (OverflowError, ValueError):
            return str(obj)
    elif isinstance(obj, pd.Series):
        return clean_for_json(obj.to_dict())
    elif isinstance(obj, pd.DataFrame):
        return clean_for_json(obj.to_dict(orient='records'))
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    return obj



# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setting matplotlib 
os.environ['MATPLOTLIB_BACKEND'] = 'Agg'
matplotlib.use('Agg')

# Loading variables
load_dotenv()

# def get_allowed_origins() -> List[str]:
#     # Get from environment variable or fall back to default
#     default_origins = [
#         "http://localhost:3000",
#         "https://matclinics-frontend.onrender.com"
#     ]
#     origins_str = os.getenv("ALLOWED_ORIGINS")
#     if origins_str:
#         try:
#             return origins_str.split(",")
#         except Exception:
#             return default_origins
#     return default_origins

def load_scripts_from_directory(rag_system, directory="scripts"):
    """Load all .py files from the scripts directory into the RAG system"""
    script_files = glob.glob(os.path.join(directory, "*.py"))
    for script_file in script_files:
        script_name = os.path.basename(script_file).replace('.py', '')
        with open(script_file, 'r') as f:
            script_content = f.read()
            # Extract docstring if it exists
            script_lines = script_content.split('\n')
            description = ""
            if '"""' in script_content:
                for line in script_lines:
                    if '"""' in line:
                        description = line.replace('"""', '').strip()
                        break
            if not description:
                description = f"Script for {script_name}"
            print(f"Loading script: {script_name} with description: {description}")  # Debug line
            rag_system.add_script(script_name, script_content, description)

app = FastAPI()

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Request headers: {request.headers}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Simple CORS configuration for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://matclinics-frontend.onrender.com"  # This exactly matches your URL
    ],
    allow_credentials=False,  # Changed to False since we're not using credentials
    allow_methods=["*"],
    allow_headers=["*"]
)

# Import your RAG system
from paste import AnalysisRAG, CodeExecutor

def get_openai_key():
    key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    if not key:
        raise ValueError("OpenAI API key not found in environment variables")
    return key

# Initialize your RAG system
rag_system = AnalysisRAG(openai_api_key=get_openai_key())

# Load scripts after initializing RAG system
load_scripts_from_directory(rag_system) 

class QueryRequest(BaseModel):
    query: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    class Config:
        json_schema_extra = {
            "example": {
                "query": "analyze data",
                "input_data": {
                    "df": [],
                    "schema": []
                }
            }
        }

@app.post("/api/analyze")
async def analyze_query(request: QueryRequest):
    try:
        print(f"Received query with schema: {request.input_data.get('schema', [])}")
        results = await rag_system.execute_analysis(
            query=request.query,
            input_data=request.input_data
        )
        # Clean and convert results before returning
        cleaned_results = clean_for_json(results)
        return json.loads(json.dumps(cleaned_results, cls=CustomJSONEncoder))
    except Exception as e:
        print(f"Error in analyze_query: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
