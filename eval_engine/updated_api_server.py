import logging
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import uvicorn
from pathlib import Path

# Import the evaluation function from our refactored module
from updated_evaluator import generate_evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Interview Evaluation API",
    description="An AI-powered API to evaluate interview transcripts and provide detailed performance assessments.",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class MessageModel(BaseModel):
    role: str = Field(..., description="The role of the message sender (ai or human)")
    content: str = Field(..., description="The content of the message")
    
    @field_validator('role')
    def validate_role(cls, v):
        if v.lower() not in ['ai', 'human']:
            raise ValueError('Role must be either "ai" or "human"')
        return v.lower()
    
    @field_validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

class EvaluationRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100, description="Unique session identifier")
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique user identifier")
    messages: List[MessageModel] = Field(..., min_items=1, description="Interview transcript messages")
    
    @field_validator('session_id', 'user_id')
    def validate_ids(cls, v):
        if not v or not v.strip():
            raise ValueError('ID cannot be empty')
        return v.strip()

class EvaluationResponse(BaseModel):
    session_id: str
    user_id: str
    summary: str = None
    problem_solving_correctness: str = None
    problem_solving_correctness_score: int = None
    code_quality_maintainability: str = None
    code_quality_maintainability_score: int = None
    technical_knowledge_depth: str = None
    technical_knowledge_depth_score: int = None
    overall_technical_score: int = None
    language_and_tool_proficiency: str = None
    problem_comprehension_and_clarification: str = None
    edge_case_handling_and_testing: str = None
    debugging_and_troubleshooting: str = None
    hints_used_independence: str = None
    communication_reasoning: str = None
    receptiveness_to_feedback: str = None

# HTML content for the frontend (embedded)
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Interview Evaluator</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/lucide/0.263.1/umd/lucide.js"></script>
    <!-- Add your CSS and HTML content here -->
</head>
<body>
    <!-- Your HTML content from the artifact -->
</body>
</html>
"""

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serves the main evaluation HTML page."""
    try:
        # Try to serve from templates directory first
        template_path = Path("templates/evaluation.html")
        if template_path.exists():
            return FileResponse(template_path)
        else:
            # Fall back to embedded HTML or serve the artifact HTML
            logger.warning("Template file not found, serving embedded HTML")
            return HTMLResponse(content=HTML_CONTENT, status_code=200)
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return HTMLResponse(
            content="<h1>AI Interview Evaluator</h1><p>Frontend temporarily unavailable. Please use the API directly.</p>",
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "AI Interview Evaluator is running"}

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_session(request: EvaluationRequest):
    """
    Receives an interview transcript and returns a structured AI evaluation.
    
    Args:
        request: EvaluationRequest containing session_id, user_id, and messages
        
    Returns:
        EvaluationResponse with detailed evaluation results
        
    Raises:
        HTTPException: If evaluation fails or required services are unavailable
    """
    try:
        logger.info(f"Received evaluation request for session: {request.session_id}, user: {request.user_id}")
        logger.debug(f"Message count: {len(request.messages)}")
        
        # Convert Pydantic models to dictionaries for the evaluator
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Validate that we have both AI and human messages
        roles = [msg.role for msg in request.messages]
        if 'ai' not in roles:
            raise HTTPException(
                status_code=400, 
                detail="Interview transcript must contain at least one AI message"
            )
        if 'human' not in roles:
            raise HTTPException(
                status_code=400, 
                detail="Interview transcript must contain at least one human message"
            )
        
        # Call the evaluation function
        evaluation_data = await generate_evaluation(
            session_id=request.session_id,
            user_id=request.user_id,
            messages=messages_dict
        )
        
        logger.info(f"Evaluation completed successfully for session: {request.session_id}")
        return evaluation_data
        
    except ConnectionError as e:
        logger.error(f"LLM Connection Error for session {request.session_id}: {e}")
        raise HTTPException(
            status_code=503, 
            detail="AI evaluation service is temporarily unavailable. Please check your API key configuration."
        )
    except ValueError as e:
        logger.error(f"Validation error for session {request.session_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during evaluation for session {request.session_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing the evaluation. Please try again."
        )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler."""
    return {
        "error": "Not Found",
        "message": f"The requested path '{request.url.path}' was not found.",
        "available_endpoints": ["/", "/evaluate", "/health", "/docs"]
    }

@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """Custom validation error handler."""
    logger.warning(f"Validation error for {request.url.path}: {exc}")
    return {
        "error": "Validation Error",
        "message": "The request data is invalid. Please check your input format.",
        "details": exc.detail if hasattr(exc, 'detail') else str(exc)
    }

if __name__ == "__main__":
    logger.info("Starting AI Interview Evaluation API server...")
    
    # Check if the evaluator can be initialized
    try:
        import os
        if not os.getenv("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY not found in environment variables")
            logger.warning("The evaluation functionality will not work without a valid API key")
    except Exception as e:
        logger.error(f"Error checking configuration: {e}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8009,
        log_level="info",
        access_log=True
    )