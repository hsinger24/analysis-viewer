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
from starlette.responses import JSONResponse 

class NumericJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return self._handle_pandas(obj)
            elif isinstance(obj, np.ndarray):
                return self._handle_numpy(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return self._handle_numpy_scalar(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
        except Exception:
            return None
            
    def _handle_pandas(self, obj):
        if isinstance(obj, pd.DataFrame):
            return {
                str(k): self._clean_value(v) 
                for k, v in obj.to_dict('index').items()
            }
        return self._clean_value(obj.to_dict())
            
    def _handle_numpy(self, obj):
        return [self._clean_value(x) for x in obj.tolist()]
        
    def _handle_numpy_scalar(self, obj):
        if np.isnan(obj) or np.isinf(obj):
            return None
        try:
            value = float(obj)
            if abs(value) > 1e308:
                return str(value)
            return value
        except:
            return str(obj)
            
    def _clean_value(self, obj):
        if pd.isna(obj) or (
            isinstance(obj, (float, np.floating)) and 
            (np.isnan(obj) or np.isinf(obj))
        ):
            return None
        if isinstance(obj, dict):
            return {str(k): self._clean_value(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._clean_value(x) for x in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return self._handle_numpy_scalar(obj)
        return obj

def render_json(obj):
    return json.dumps(obj, cls=NumericJSONEncoder)



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
        results = await rag_system.execute_analysis(
            query=request.query,
            input_data=request.input_data
        )
        # Remove double encoding
        return JSONResponse(
            content=json.loads(render_json(results)), 
            media_type="application/json"
        )
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
