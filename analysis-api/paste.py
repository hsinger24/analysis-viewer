# Imports

import pandas as pd
import numpy as np
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import json
import ast
import sys
from io import StringIO
import contextlib
import traceback
from typing import Any, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv
import matplotlib

# Setting matplotlib 
os.environ['MATPLOTLIB_BACKEND'] = 'Agg'
matplotlib.use('Agg')

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


# Loading variables
load_dotenv()

def get_openai_key():
    key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    if not key:
        raise ValueError("OpenAI API key not found in environment variables")
    return key

def generate_sample_data(n_patients=100, days=90):
    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(days)]
    
    # Generate patient visits (multiple per patient)
    patient_ids = list(range(1, n_patients + 1))
    records = []
    
    medications = ['Methadone', 'Buprenorphine', 'Naltrexone']
    
    for patient_id in patient_ids:
        # Random number of visits for each patient (2-6 visits)
        n_visits = np.random.randint(2, 7)
        visit_dates = sorted(np.random.choice(dates, n_visits, replace=False))
        
        for visit_date in visit_dates:
            next_appointment = visit_date + timedelta(days=np.random.randint(14, 35))
            records.append({
                'patient_id': patient_id,
                'medication_name': np.random.choice(medications),
                'prescription_date': visit_date,
                'dosage': np.random.normal(50, 10),
                'next_appointment': next_appointment,
                'attendance_status': np.random.choice(['Attended', 'Missed'], p=[0.8, 0.2])
            })
    
    return pd.DataFrame(records)



# Script execution

class CodeExecutor:
    def __init__(self):
        """Initialize with proper imports in the global namespace"""
        # First import all required modules
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime, timedelta

        # Store them in the globals dict
        self.globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'datetime': datetime,
            'timedelta': timedelta,
            'validate_schema': self.validate_schema
        }
        self.locals = {}
    
    def validate_schema(self, required_columns: list, available_columns: list) -> tuple:
        """
        Helper function to validate if required columns are available
        Returns (bool, str) indicating success and error message if any
        """
        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        return True, ""
        
    @contextlib.contextmanager
    def capture_output(self):
        """Capture stdout and stderr"""
        stdout, stderr = StringIO(), StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = stdout, stderr
            yield stdout, stderr
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            
    def execute_code(self, code: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute code with proper namespace management"""
        if input_data:
            # Update locals with input data
            self.locals.update(input_data)
            
        # Ensure globals are available in locals
        execution_namespace = {**self.globals, **self.locals}
            
        with self.capture_output() as (stdout, stderr):
            try:
                exec(code, execution_namespace, execution_namespace)
                output = stdout.getvalue()
                errors = stderr.getvalue()
                
                # Convert any matplotlib figures to base64
                import io
                import base64
                for k, v in execution_namespace.items():
                    if str(type(v).__name__) == 'Figure':
                        buf = io.BytesIO()
                        v.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        execution_namespace[k] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
                        buf.close()
                
                # Update locals with new definitions
                self.locals.update({
                    k: v for k, v in execution_namespace.items() 
                    if not k.startswith('__') and 
                    k not in self.globals and 
                    k not in (input_data or {})
                })
                
                return {
                    'success': True,
                    'output': output,
                    'errors': errors,
                    'results': self.locals
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'output': stdout.getvalue(),
                    'errors': stderr.getvalue()
                }







# Script Generation

class AnalysisRAG:

    def __init__(self, openai_api_key):
        """
        Initialize the RAG system with OpenAI API key and storage for scripts and data
        """
        self.client = openai.OpenAI(api_key=get_openai_key())
        self.vectorizer = TfidfVectorizer()
        self.scripts = {}  # Store scripts and their descriptions
        self.script_embeddings = None
        self.executor = CodeExecutor()  # Add code executor
        
    def add_script(self, script_name: str, script_content: str, description: str):
        """
        Add a new analysis script to the RAG system
        """
        self.scripts[script_name] = {
            'content': script_content,
            'description': description
        }
        # Rebuild embeddings when new script is added
        self._update_embeddings()
        
    def _update_embeddings(self):
        """
        Update embeddings for all scripts using TF-IDF
        """
        descriptions = [script['description'] for script in self.scripts.values()]
        self.script_embeddings = self.vectorizer.fit_transform(descriptions)
        
    def save_system(self, filepath: str):
        """
        Save the RAG system to disk
        """
        system_data = {
            'scripts': self.scripts,
            'vectorizer': self.vectorizer
        }
        with open(filepath, 'wb') as f:
            pickle.dump(system_data, f)
            
    def load_system(self, filepath: str):
        """
        Load the RAG system from disk
        """
        with open(filepath, 'rb') as f:
            system_data = pickle.load(f)
        self.scripts = system_data['scripts']
        self.vectorizer = system_data['vectorizer']
        self._update_embeddings()
        
    def query(self, query: str, n_results: int = 3) -> list:
        """
        Query the RAG system to find relevant analysis scripts
        """
        # Transform query using the same vectorizer
        query_embedding = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_embedding, self.script_embeddings).flatten()
        
        # Get top N most similar scripts
        top_indices = similarities.argsort()[-n_results:][::-1]
        
        results = []
        for idx in top_indices:
            script_name = list(self.scripts.keys())[idx]
            similarity_score = similarities[idx]
            results.append({
                'script_name': script_name,
                'score': similarity_score,
                'content': self.scripts[script_name]['content'],
                'description': self.scripts[script_name]['description']
            })
            
        return results
    
    def get_ai_response(self, query: str, context: list, schema: list = None) -> dict:
        """
        Get AI-generated response with executable code snippets
        Modified to ensure proper code generation format and handling
        """
        context_text = ""
        for idx, script in enumerate(context, 1):
            context_text += f"\nScript {idx}: {script['script_name']}\n"
            context_text += f"Description: {script['description']}\n"
            context_text += f"Content:\n{script['content']}\n"
        
        # Add schema information to the prompt
        schema_info = ""
        if schema:
            schema_info = f"\nAvailable columns in data: {', '.join(schema)}\n"

        system_message = """You are an expert data analysis assistant. Your task is to:
        1. Generate EXECUTABLE Python code snippets for the analysis
        2. Include complete function definitions and implementations
        3. Write clean, properly formatted code without type hints
        4. Use only basic Python syntax features
        5. Ensure all string literals use single quotes
        6. Avoid line continuation characters (\\)
        7. Keep all code on single lines or use proper indentation
        8. Use parentheses for line breaks instead of backslashes
        9. Check if required columns are available in the data schema

        IMPORTANT FORMATTING RULES:
        - Verify required columns exist before analysis
        - Handle missing columns gracefully
        - Include column validation in generated code
        - Return clear error messages if required columns are missing
        - NO type hints or annotations
        - NO arrow syntax (->)
        - NO line continuation characters (\\)
        - Use parentheses for long lines
        - Keep function definitions simple

        The execution_code must explicitly call each function with 'df' as input and include datetime if needed, like this:
        transitions = analyze_medication_transitions(df)
        risks = calculate_risk_metrics(df)
        summary = generate_summary_report(df, datetime(2024, 1, 1), datetime(2024, 3, 1))
        trend_plot = plot_trends(df, 'prescriptions')

        Example of correct formatting:

        def analyze_data(df):
            # Use parentheses for long lines
            result = (df.groupby('column')
                       .agg({'value': 'mean'})
                       .reset_index())
            return result

        Return your response in this exact JSON format:
        {
            "explanation": "Detailed explanation of what the code will do",
            "setup_code": "# Imports and initialization code",
            "code_snippets": [
                {
                    "function_name": "function_name",
                    "code": "# Complete function definition",
                    "inputs": "description of required inputs",
                    "outputs": "description of outputs"
                }
            ],
            "execution_code": "# Code that calls the functions"
        }"""

        user_message = f"""Query: {query}
        {schema_info}
        Available code from context:
        {context_text}

        Generate complete, executable code following the formatting rules exactly."""

        client = openai.OpenAI(api_key=get_openai_key())
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={ "type": "json_object" }
        )	

        return json.loads(response.choices[0].message.content)

    async def execute_analysis(self, query: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        print("Starting analysis...")
        input_data = input_data or {}
        schema = input_data.get('schema', [])
        df_data = input_data.get('df', [])
        
        # Convert data to DataFrame with proper date handling
        try:
            if isinstance(df_data, list) and df_data:
                # Convert date strings to datetime objects
                for row in df_data:
                    for key, value in row.items():
                        # Convert any date-like strings to datetime
                        if isinstance(value, str):
                            try:
                                if 'date' in key.lower() or 'time' in key.lower():
                                    row[key] = pd.to_datetime(value)
                            except (ValueError, TypeError):
                                pass  # Keep original value if conversion fails
                
                # Convert list to DataFrame
                df_data = pd.DataFrame(df_data)
                
                # Ensure numeric columns are properly typed
                for col in df_data.columns:
                    if col == 'patient_id':
                        df_data[col] = pd.to_numeric(df_data[col], errors='ignore')
                    elif col == 'dosage':
                        df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
        except Exception as e:
            print(f"Error converting data to DataFrame: {str(e)}")
            return {
                'explanation': "Error converting input data to proper format",
                'setup': {'success': False, 'error': str(e)}
            }
        
        context = self.query(query)
        ai_response = self.get_ai_response(query, context, schema)
        print("AI Response received:", ai_response)

        results = {
            'explanation': ai_response['explanation'],
            'steps': []
        }

        if 'setup_code' in ai_response:
            setup_code = ai_response['setup_code'].strip()
            if setup_code:
                setup_result = self.executor.execute_code(
                    setup_code,
                    {'df': df_data, 'schema': schema}
                )
                results['setup'] = {
                    'success': setup_result['success'],
                    'error': setup_result.get('error', None)
                }
                if not setup_result['success']:
                    return results

                # Update input_data with setup results
                if setup_result['success'] and setup_result['results']:
                    input_data.update(setup_result['results'])

        # Execute each function definition
        defined_functions = []
        for snippet in ai_response['code_snippets']:
            function_code = snippet['code'].strip()
            function_name = snippet['function_name']

            # Define the function
            function_result = self.executor.execute_code(
                function_code,
                {'df': df_data, 'schema': schema, **input_data}
            )

            if function_result['success']:
                # Execute the function immediately
                execution = self.executor.execute_code(
                    f"result = {function_name}(df)",
                    {'df': df_data, **function_result['results']}
                )
                
                step_result = {
                    'name': function_name,
                    'success': True,
                    'output': execution.get('output', ''),
                    'results': execution.get('results', {}).get('result', None)
                }
                input_data.update(execution.get('results', {}))
            else:
                step_result = {
                    'name': function_name,
                    'success': False,
                    'error': function_result.get('error', None)
                }

            results['steps'].append(step_result)
            if not function_result['success']:
                return results

        # Execute the final execution code
        if 'execution_code' in ai_response:
            execution_code = ai_response['execution_code'].strip()
            if execution_code:
                execution_result = self.executor.execute_code(
                    execution_code,
                    {
                        'df': df_data, 
                        'schema': schema, 
                        'datetime': datetime,
                        **input_data
                    }
                )

                results['execution'] = {
                    'success': execution_result['success'],
                    'error': execution_result.get('error', None),
                    'output': execution_result.get('output', '').strip()
                }

                if execution_result['success'] and 'results' in execution_result:
                    # Clean up results for display
                    cleaned_results = {}
                    for key, value in execution_result['results'].items():
                        if callable(value):
                            cleaned_results[key] = f"<Function '{key}' defined>"
                        elif isinstance(value, (pd.DataFrame, pd.Series)):
                            cleaned_results[key] = f"<DataFrame with shape {value.shape}>"
                        elif str(type(value).__name__) == 'Figure':
                            cleaned_results[key] = value  # Keep the base64 image as is
                        else:
                            # Clean any non-JSON-compliant values
                            cleaned_results[key] = clean_for_json(value)
                    results['execution']['results'] = cleaned_results
        
         # Add the debug section HERE, right before the return statement
        print("DEBUG: Examining results structure")
        for key, value in results.items():
            print(f"Key: {key}")
            if key == 'steps':
                for step in value:
                    print(f"  Step: {step['name']}")
                    if 'results' in step and step['results'] is not None:
                        print(f"    Results type: {type(step['results'])}")
                        if isinstance(step['results'], dict):
                            for k, v in step['results'].items():
                                print(f"      {k}: {type(v)}")
            elif key == 'execution' and value and 'results' in value:
                for k, v in value['results'].items():
                    print(f"  Execution result {k}: {type(v)}")

        return clean_for_json(results)



