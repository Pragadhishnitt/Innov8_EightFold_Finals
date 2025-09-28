"""
CodeSage MVP FastAPI Backend
Simplified API without sessions for direct CodeSage integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time
from datetime import datetime

# Import your CodeSage system
from codesage_system import (
    CodeSageInterview, 
    AgentState, 
    initial_question_setter,
    DatabaseService,
    supabase  # Import supabase client
)

# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class StartInterviewRequest(BaseModel):
    """Request model for starting an interview"""
    candidate_name: str
    difficulty: str  # "easy", "medium", "hard"
    topic: str       # "Array", "String", "Graph", etc.

class StartInterviewResponse(BaseModel):
    """Response model for starting an interview"""
    status: str
    message: str
    initial_question: str
    question_id: int
    question_title: str
    difficulty: str
    topic: str

class UpdateCodeRequest(BaseModel):
    """Request model for updating code from sandbox"""
    question_id: int
    difficulty: str
    code_content: str
    candidate_name: Optional[str] = "Candidate"

class UpdateCodeResponse(BaseModel):
    """Response model for code updates"""
    status: str
    ai_response: Optional[str] = None
    action_type: str  # "hint", "follow_up", "next_question", "evaluation", "response"
    should_respond: bool = False  # Whether AI has something to say

class UpdateTranscriptRequest(BaseModel):
    """Request model for updating voice transcript"""
    question_id: int
    difficulty: str
    transcript_text: str
    candidate_name: Optional[str] = "Candidate"

class UpdateTranscriptResponse(BaseModel):
    """Response model for transcript updates"""
    status: str
    ai_response: Optional[str] = None
    action_type: str
    should_respond: bool = False

class ProcessInterviewRequest(BaseModel):
    """Request model for processing interview step with both code and transcript"""
    question_id: int
    difficulty: str
    code_content: str
    transcript_text: str
    candidate_name: Optional[str] = "Candidate"

class ProcessInterviewResponse(BaseModel):
    """Response model for interview processing"""
    status: str
    ai_response: str
    action_type: str  # "hint", "follow_up", "next_question", "evaluation", "response"
    next_question: Optional[str] = None
    next_question_id: Optional[int] = None
    interview_complete: bool = False

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(
    title="CodeSage MVP Interview API",
    description="Simplified backend API for CodeSage technical interviews",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CODESAGE INTEGRATION
# =============================================================================

class CodeSageDirectIntegration:
    """Direct integration with CodeSage system without sessions"""
    
    @staticmethod
    def debug_database_search(difficulty: str, topic: str):
        """Debug function to check database search"""
        try:
            print(f"\n=== DATABASE SEARCH DEBUG ===")
            print(f"Searching for: difficulty='{difficulty}', topic='{topic}'")
            
            # Check what's in the database first
            all_questions_response = supabase.table('questions').select('id, title, difficulty, topics').execute()
            print(f"Total questions in database: {len(all_questions_response.data) if all_questions_response.data else 0}")
            
            if all_questions_response.data:
                print("Available questions:")
                for q in all_questions_response.data[:5]:  # Show first 5
                    print(f"  ID: {q['id']}, Title: {q['title']}, Difficulty: {q['difficulty']}, Topics: {q.get('topics', 'N/A')}")
                
            # Now try the actual search
            print(f"\nSearching for difficulty='{difficulty}' AND topics contains '{topic}':")
            search_response = supabase.table('questions').select('id, title, difficulty, topics') \
                .eq('difficulty', difficulty) \
                .cs('topics', [topic]) \
                .execute()
            
            print(f"Search results: {len(search_response.data) if search_response.data else 0} matches")
            if search_response.data:
                for result in search_response.data:
                    print(f"  Found: {result}")
            
            print("=== END DEBUG ===\n")
            
            return search_response.data[0]['id'] if search_response.data else None
            
        except Exception as e:
            print(f"Debug search error: {e}")
            return None

    @staticmethod
    def create_initial_state(candidate_name: str, difficulty: str, topic: str) -> tuple[AgentState, str, int, str]:
        """Create initial interview state and get first question from database"""
        print(f"Creating initial state for {candidate_name}")
        print(f"Searching database for: difficulty='{difficulty}', topic='{topic}'")
        
        # Use debug function to see what's happening
        question_id = CodeSageDirectIntegration.debug_database_search(difficulty, topic)
        
        if not question_id:
            print(f"No exact match found. Trying fallback search for difficulty='{difficulty}' only...")
            # Try to find any question with just the difficulty
            try:
                response = supabase.table('questions').select('id, title, difficulty').eq('difficulty', difficulty).execute()
                if response.data:
                    question_id = response.data[0]['id']
                    print(f"Found fallback question: {response.data[0]['title']} (ID: {question_id})")
                else:
                    print("No questions found for difficulty, using absolute fallback")
                    question_id = 101  # Absolute fallback
            except Exception as e:
                print(f"Database fallback search failed: {e}")
                question_id = 101
        
        print(f"Final selected question ID: {question_id}")
        
        # Get question details from database
        question_details = DatabaseService.get_question(question_id)
        
        if "error" in question_details:
            print(f"Error fetching question {question_id}: {question_details['error']}")
            # Use absolute fallback
            question_details = {
                "title": "Two Sum",
                "problem_statement": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                "difficulty": difficulty
            }
        
        question_text = question_details.get("problem_statement", "Question not found")
        question_title = question_details.get("title", "Unknown Problem")
        actual_difficulty = question_details.get("difficulty", difficulty)
        
        print(f"Question details: {question_title} ({actual_difficulty})")
        
        start_time = time.time()
        
        initial_state = AgentState(
            messages=[{
                "role": "assistant",
                "content": f"Welcome {candidate_name}! I'm your AI technical interviewer. Let's start with this problem:\n\n**{question_title} ({actual_difficulty.upper()})**\n\n{question_text}",
                "timestamp": start_time
            }],
            question_id=[question_id],
            difficulty=[actual_difficulty],
            code_script="",
            voice_transcript="",
            start_time=start_time,
            question_start_time=start_time,
            time_exceeded=False,
            interview_complete=False,
            hints_used=0,
            performance_score=0.0
        )
        
        return initial_state, question_text, question_id, question_title
    
    @staticmethod
    def process_interview_step(question_id: int, difficulty: str, code_content: str, 
                             transcript_text: str, candidate_name: str = "Candidate") -> Dict[str, Any]:
        """Process one interview step with current code and transcript"""
        
        try:
            # Create CodeSage interview instance
            interview_system = CodeSageInterview()
            
            # Create current state based on inputs
            current_time = time.time()
            current_state = AgentState(
                messages=[{
                    "role": "assistant",
                    "content": f"Analyzing your current progress...",
                    "timestamp": current_time
                }],
                question_id=[question_id],
                difficulty=[difficulty],
                code_script=code_content,
                voice_transcript=transcript_text,
                start_time=current_time - 300,  # Simulate 5 min elapsed
                question_start_time=current_time - 300,
                time_exceeded=False,
                interview_complete=False,
                hints_used=0,
                performance_score=0.0
            )
            
            print(f"Processing interview step for question {question_id}")
            print(f"Code length: {len(code_content)}")
            print(f"Transcript length: {len(transcript_text)}")
            
            # Run the CodeSage graph with current state
            updated_state = interview_system.graph.invoke(current_state)
            
            # Extract AI response from messages
            messages = updated_state.get("messages", [])
            if messages:
                latest_message = messages[-1]
                ai_response = latest_message.get("content", "")
                
                # Determine action type based on response content
                action_type = "response"
                next_question = None
                next_question_id = None
                interview_complete = updated_state.get("interview_complete", False)
                
                if "Hint" in ai_response or "💡" in ai_response:
                    action_type = "hint"
                elif "complexity" in ai_response.lower() or "🔍" in ai_response:
                    action_type = "follow_up"
                elif "Next Question" in ai_response or "📝" in ai_response:
                    action_type = "next_question"
                    # Extract next question details if available
                    new_question_ids = updated_state.get("question_id", [])
                    if len(new_question_ids) > 1:
                        next_question_id = new_question_ids[-1]
                        question_details = DatabaseService.get_question(next_question_id)
                        next_question = question_details.get("problem_statement", "")
                elif "EVALUATION" in ai_response or "📊" in ai_response:
                    action_type = "evaluation"
                    interview_complete = True
                
                return {
                    "status": "success",
                    "ai_response": ai_response,
                    "action_type": action_type,
                    "next_question": next_question,
                    "next_question_id": next_question_id,
                    "interview_complete": interview_complete
                }
            else:
                return {
                    "status": "error",
                    "ai_response": "No response generated from CodeSage system",
                    "action_type": "error",
                    "next_question": None,
                    "next_question_id": None,
                    "interview_complete": False
                }
                
        except Exception as e:
            print(f"Error in CodeSage processing: {e}")
            return {
                "status": "error", 
                "ai_response": f"System error: {str(e)}",
                "action_type": "error",
                "next_question": None,
                "next_question_id": None,
                "interview_complete": False
            }

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "CodeSage MVP API is running!", "status": "healthy"}

@app.post("/api/interview/start", response_model=StartInterviewResponse)
async def start_interview(request: StartInterviewRequest):
    """
    Start a new interview, get initial question from database, and start interview agent
    """
    try:
        print(f"Starting interview for {request.candidate_name}")
        print(f"Requested: {request.difficulty} difficulty, {request.topic} topic")
        
        # Create initial state with proper database integration
        initial_state, question_text, question_id, question_title = CodeSageDirectIntegration.create_initial_state(
            request.candidate_name,
            request.difficulty,
            request.topic
        )
        
        # Actually start the interview agent to process the initial state
        interview_system = CodeSageInterview()
        print("Starting CodeSage interview agent...")
        
        # Run one iteration of the interview graph to start the interview
        try:
            processed_state = interview_system.graph.invoke(initial_state)
            
            # Get the latest message from the interview agent
            messages = processed_state.get("messages", [])
            if messages and len(messages) > 1:
                # Use the agent's response instead of just the initial question
                latest_response = messages[-1].get("content", question_text)
                print("Interview agent started successfully")
            else:
                latest_response = question_text
                print("Using initial question as fallback")
                
        except Exception as agent_error:
            print(f"Interview agent start error: {agent_error}")
            latest_response = question_text
        
        print(f"Selected question: {question_title} (ID: {question_id})")
        
        return StartInterviewResponse(
            status="success",
            message=f"Interview started for {request.candidate_name}",
            initial_question=latest_response,
            question_id=question_id,
            question_title=question_title,
            difficulty=initial_state["difficulty"][-1],  # Use actual difficulty from database
            topic=request.topic
        )
        
    except Exception as e:
        print(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")

@app.post("/api/interview/update-code", response_model=UpdateCodeResponse)
async def update_code(request: UpdateCodeRequest):
    """
    Update code from sandbox and potentially get AI response
    Called whenever user types/modifies code in the editor
    """
    try:
        print(f"Code update for question {request.question_id}")
        print(f"Code length: {len(request.code_content)} characters")
        print(f"Code preview: {request.code_content[:100]}...")
        
        # Create current state with just the code (no transcript for this endpoint)
        current_time = time.time()
        current_state = AgentState(
            messages=[{
                "role": "assistant",
                "content": f"Analyzing your code...",
                "timestamp": current_time
            }],
            question_id=[request.question_id],
            difficulty=[request.difficulty],
            code_script=request.code_content,
            voice_transcript="",  # Empty transcript for code-only update
            start_time=current_time - 300,
            question_start_time=current_time - 300,
            time_exceeded=False,
            interview_complete=False,
            hints_used=0,
            performance_score=0.0
        )
        
        # Process through CodeSage system
        interview_system = CodeSageInterview()
        
        # Only run interview agent if there's substantial code (> 20 characters)
        # This prevents constant AI responses for every keystroke
        should_process = len(request.code_content.strip()) > 20
        
        if should_process:
            print("Processing code through CodeSage agents...")
            try:
                updated_state = interview_system.graph.invoke(current_state)
                
                messages = updated_state.get("messages", [])
                if messages and len(messages) > 1:
                    latest_message = messages[-1]
                    ai_response = latest_message.get("content", "")
                    
                    # Determine if this is worth responding to
                    action_type = "code_analysis"
                    should_respond = True
                    
                    if "Hint" in ai_response or "💡" in ai_response:
                        action_type = "hint"
                    elif "complexity" in ai_response.lower() or "🔍" in ai_response:
                        action_type = "follow_up"
                    elif len(ai_response.strip()) < 20:
                        should_respond = False  # Don't respond to very short messages
                    
                    print(f"AI Response (Code): {action_type}")
                    
                    return UpdateCodeResponse(
                        status="success",
                        ai_response=ai_response,
                        action_type=action_type,
                        should_respond=should_respond
                    )
                else:
                    print("No substantial AI response for code update")
                    return UpdateCodeResponse(
                        status="success",
                        ai_response="",
                        action_type="no_response",
                        should_respond=False
                    )
                    
            except Exception as agent_error:
                print(f"CodeSage agent error: {agent_error}")
                return UpdateCodeResponse(
                    status="success",
                    ai_response="",
                    action_type="no_response", 
                    should_respond=False
                )
        else:
            print("Code too short, not processing through agents")
            return UpdateCodeResponse(
                status="success",
                ai_response="",
                action_type="no_response",
                should_respond=False
            )
            
    except Exception as e:
        print(f"Error processing code update: {e}")
        raise HTTPException(status_code=500, detail=f"Code processing failed: {str(e)}")

@app.post("/api/interview/update-transcript", response_model=UpdateTranscriptResponse)
async def update_transcript(request: UpdateTranscriptRequest):
    """
    Update voice transcript and get AI response
    Called when speech-to-text provides new transcript data
    """
    try:
        print(f"Transcript update for question {request.question_id}")
        print(f"Transcript: '{request.transcript_text}'")
        
        # Create current state with just the transcript (no code for this endpoint)
        current_time = time.time()
        current_state = AgentState(
            messages=[{
                "role": "user",
                "content": request.transcript_text,
                "timestamp": current_time
            }],
            question_id=[request.question_id],
            difficulty=[request.difficulty],
            code_script="",  # Empty code for transcript-only update
            voice_transcript=request.transcript_text,
            start_time=current_time - 300,
            question_start_time=current_time - 300,
            time_exceeded=False,
            interview_complete=False,
            hints_used=0,
            performance_score=0.0
        )
        
        # Always process transcript through CodeSage system
        # Voice input usually indicates the candidate wants interaction
        print("Processing transcript through CodeSage agents...")
        
        try:
            interview_system = CodeSageInterview()
            updated_state = interview_system.graph.invoke(current_state)
            
            messages = updated_state.get("messages", [])
            if messages and len(messages) > 1:
                latest_message = messages[-1]
                ai_response = latest_message.get("content", "")
                
                action_type = "voice_response"
                
                if "Hint" in ai_response or "💡" in ai_response:
                    action_type = "hint"
                elif "complexity" in ai_response.lower() or "🔍" in ai_response:
                    action_type = "follow_up"
                elif "Next Question" in ai_response or "📝" in ai_response:
                    action_type = "next_question"
                elif "EVALUATION" in ai_response or "📊" in ai_response:
                    action_type = "evaluation"
                
                print(f"AI Response (Voice): {action_type}")
                
                return UpdateTranscriptResponse(
                    status="success",
                    ai_response=ai_response,
                    action_type=action_type,
                    should_respond=True
                )
            else:
                print("No AI response generated for transcript")
                return UpdateTranscriptResponse(
                    status="success",
                    ai_response="I'm listening. Please continue or let me know if you need help.",
                    action_type="acknowledgment",
                    should_respond=True
                )
                
        except Exception as agent_error:
            print(f"CodeSage agent error: {agent_error}")
            return UpdateTranscriptResponse(
                status="success",
                ai_response="I heard you. Could you elaborate on that?",
                action_type="fallback",
                should_respond=True
            )
            
    except Exception as e:
        print(f"Error processing transcript: {e}")
        raise HTTPException(status_code=500, detail=f"Transcript processing failed: {str(e)}")

@app.post("/api/interview/process", response_model=ProcessInterviewResponse)
async def process_interview(request: ProcessInterviewRequest):
    """
    Process current interview state (code + transcript) through CodeSage system
    Returns AI response that should be displayed on frontend
    """
    try:
        print(f"Processing interview for question {request.question_id}")
        print(f"Code: {len(request.code_content)} chars, Transcript: {len(request.transcript_text)} chars")
        
        # Process through CodeSage system
        result = CodeSageDirectIntegration.process_interview_step(
            request.question_id,
            request.difficulty,
            request.code_content,
            request.transcript_text,
            request.candidate_name
        )
        
        print(f"AI Response Action: {result['action_type']}")
        print(f"Response preview: {result['ai_response'][:100]}...")
        
        return ProcessInterviewResponse(
            status=result["status"],
            ai_response=result["ai_response"],
            action_type=result["action_type"],
            next_question=result["next_question"],
            next_question_id=result["next_question_id"],
            interview_complete=result["interview_complete"]
        )
        
    except Exception as e:
        print(f"Error processing interview: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/question/{question_id}")
async def get_question_details(question_id: int):
    """
    Get detailed information about a specific question
    Useful when frontend needs to display question details
    """
    try:
        question_details = DatabaseService.get_question(question_id)
        
        if "error" in question_details:
            raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
        
        return {
            "status": "success",
            "question": question_details
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching question details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hints/{question_id}")
async def get_question_hints(question_id: int):
    """
    Get available hints for a question
    Frontend can use this to show hint availability
    """
    try:
        hints = DatabaseService.get_hints(question_id)
        
        return {
            "status": "success",
            "hints": hints,
            "hint_count": len(hints)
        }
        
    except Exception as e:
        print(f"Error fetching hints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# STARTUP CONFIGURATION
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("🚀 CodeSage MVP API starting up...")
    print("📡 Available endpoints:")
    print("   POST /api/interview/start - Start new interview")
    print("   POST /api/interview/update-code - Update code from sandbox")
    print("   POST /api/interview/update-transcript - Update voice transcript")
    print("   POST /api/interview/process - Process combined code+transcript")
    print("   GET  /api/question/{question_id} - Get question details")
    print("   GET  /api/hints/{question_id} - Get question hints")

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
Frontend Integration Examples:

1. Start Interview:
POST /api/interview/start
{
  "candidate_name": "Alice Johnson",
  "difficulty": "hard",
  "topic": "Tree"
}

Response:
{
  "status": "success",
  "message": "Interview started for Alice Johnson",
  "initial_question": "Given a binary tree...",
  "question_id": 301,
  "question_title": "Binary Tree Traversal",
  "difficulty": "hard",
  "topic": "Tree"
}

2. Update Code from Sandbox (called on code editor changes):
POST /api/interview/update-code
{
  "question_id": 301,
  "difficulty": "hard",
  "code_content": "def inorder_traversal(root):\n    if not root:\n        return []",
  "candidate_name": "Alice"
}

Response:
{
  "status": "success",
  "ai_response": "Good start! You're handling the base case correctly. What's your approach for the recursive case?",
  "action_type": "follow_up",
  "should_respond": true
}

3. Update Voice Transcript (called when speech-to-text updates):
POST /api/interview/update-transcript
{
  "question_id": 301,
  "difficulty": "hard", 
  "transcript_text": "I think I need to use recursion to traverse left, process current, then traverse right",
  "candidate_name": "Alice"
}

Response:
{
  "status": "success",
  "ai_response": "Exactly! That's the classic inorder traversal approach. Can you implement that logic in your code?",
  "action_type": "voice_response",
  "should_respond": true
}

4. Process Combined State (optional, for complex analysis):
POST /api/interview/process
{
  "question_id": 301,
  "difficulty": "hard",
  "code_content": "def inorder_traversal(root):\n    result = []\n    def dfs(node):\n        if node:\n            dfs(node.left)\n            result.append(node.val)\n            dfs(node.right)\n    dfs(root)\n    return result",
  "transcript_text": "I implemented the recursive solution with a helper function",
  "candidate_name": "Alice"
}

Response:
{
  "status": "success",
  "ai_response": "Excellent solution! Can you analyze the time and space complexity?",
  "action_type": "follow_up",
  "next_question": null,
  "next_question_id": null,
  "interview_complete": false
}

Frontend Implementation Strategy:
- Call /api/interview/start once at the beginning
- Call /api/interview/update-code on code editor changes (debounced)
- Call /api/interview/update-transcript when new speech is transcribed
- Display ai_response in chat when should_respond is true
- Use /api/interview/process for comprehensive analysis when needed

Smart Response Logic:
- update-code only responds for substantial code (>20 chars)
- update-transcript always tries to respond (voice indicates interaction intent)
- Frontend can choose whether to display responses based on should_respond flag

"""