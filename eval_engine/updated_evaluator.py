import os
import asyncio
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- 1. Enhanced Pydantic Schema with Validation ---
class EvaluatorAgent(BaseModel):
    """Defines the structured output for an interview evaluation with comprehensive validation."""
    
    # --- Identifiers ---
    session_id: Optional[str] = Field(default=None, description="Unique identifier for the interview session.")
    user_id: Optional[str] = Field(default=None, description="Unique identifier of the candidate.")
    
    # --- Qualitative Assessment Fields ---
    problem_solving_correctness: Optional[str] = Field(
        default=None, 
        description="Assessment of problem-solving approach and correctness"
    )
    code_quality_maintainability: Optional[str] = Field(
        default=None, 
        description="Evaluation of code quality, readability, and maintainability"
    )
    technical_knowledge_depth: Optional[str] = Field(
        default=None, 
        description="Assessment of technical knowledge and understanding"
    )
    language_and_tool_proficiency: Optional[str] = Field(
        default=None, 
        description="Evaluation of programming language and tool proficiency"
    )
    problem_comprehension_and_clarification: Optional[str] = Field(
        default=None, 
        description="Assessment of problem understanding and clarification skills"
    )
    edge_case_handling_and_testing: Optional[str] = Field(
        default=None, 
        description="Evaluation of edge case consideration and testing approach"
    )
    debugging_and_troubleshooting: Optional[str] = Field(
        default=None, 
        description="Assessment of debugging and problem resolution skills"
    )
    hints_used_independence: Optional[str] = Field(
        default=None, 
        description="Evaluation of independence and hint usage"
    )
    communication_reasoning: Optional[str] = Field(
        default=None, 
        description="Assessment of communication skills and reasoning explanation"
    )
    receptiveness_to_feedback: Optional[str] = Field(
        default=None, 
        description="Evaluation of openness to feedback and adaptability"
    )
    summary: Optional[str] = Field(
        default=None, 
        description="Comprehensive summary of the candidate's performance"
    )

    # --- Numeric Score Fields with Validation ---
    problem_solving_correctness_score: Optional[int] = Field(
        default=None, 
        ge=1, le=5, 
        description="Numeric score (1-5) for problem-solving skills"
    )
    code_quality_maintainability_score: Optional[int] = Field(
        default=None, 
        ge=1, le=5, 
        description="Numeric score (1-5) for code quality"
    )
    technical_knowledge_depth_score: Optional[int] = Field(
        default=None, 
        ge=1, le=5, 
        description="Numeric score (1-5) for technical knowledge depth"
    )
    overall_technical_score: Optional[int] = Field(
        default=None, 
        ge=1, le=5, 
        description="Holistic technical score (1-5)"
    )

    @field_validator('summary')
    def validate_summary(cls, v):
        if v and len(v.strip()) < 20:
            logger.warning("Summary is too short, should be more descriptive")
        return v

    @field_validator('problem_solving_correctness_score', 'code_quality_maintainability_score', 
              'technical_knowledge_depth_score', 'overall_technical_score')
    def validate_scores(cls, v):
        if v is not None:
            if not isinstance(v, int) or not (1 <= v <= 5):
                logger.warning(f"Invalid score value: {v}, should be between 1-5")
                return None
        return v

# --- 2. Enhanced LangChain Setup ---
parser = PydanticOutputParser(pydantic_object=EvaluatorAgent)

def initialize_llm():
    """Initialize the LLM with proper error handling."""
    if not google_api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        raise ConnectionError("Google API key is required but not found in environment variables")
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", 
            google_api_key=google_api_key, 
            temperature=0.1,  # Slightly higher for more varied responses
            max_retries=3,
            request_timeout=60
        )
        logger.info("Gemini LLM initialized successfully for evaluator")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise ConnectionError(f"Failed to initialize AI model: {str(e)}")

# Initialize LLM
try:
    llm = initialize_llm()
except ConnectionError as e:
    llm = None
    logger.warning(f"LLM initialization failed: {e}")

# Enhanced prompt with better instructions and examples
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert technical interviewer and talent evaluator with years of experience in assessing software engineering candidates. 

Your task is to analyze interview transcripts and provide comprehensive, fair, and actionable evaluations. Follow these guidelines:

1. **Scoring Scale (1-5):**
   - 1 (Poor): Significant deficiencies, incorrect approaches
   - 2 (Below Average): Some understanding but major gaps
   - 3 (Average): Adequate performance with room for improvement  
   - 4 (Good): Solid performance with minor areas for improvement
   - 5 (Excellent): Outstanding performance, demonstrates mastery

2. **Assessment Focus:**
   - Look for problem-solving approach, not just correct answers
   - Value clear communication and reasoning
   - Consider code quality, edge cases, and best practices
   - Assess adaptability and learning from feedback

3. **Output Requirements:**
   - Provide specific, constructive feedback
   - Include both strengths and areas for improvement
   - Ensure scores align with qualitative assessments
   - Write a comprehensive summary (minimum 50 words)

Respond with a valid JSON object that strictly follows the provided schema. Do not include markdown formatting or additional text.

{format_instructions}"""),
    
    ("ai", "Let's work on a coding problem. Given a sorted array, find the index of a target value using binary search."),
    
    ("human", "Understood. Before I start, can I clarify a few things? What should I return if the target isn't found? Are there duplicate values? Also, I'm thinking binary search would be optimal here since the array is sorted, giving us O(log n) time complexity."),
    
    ("ai", "Great questions! Return -1 if not found, and assume no duplicates. Your approach sounds perfect."),
    
    ("human", """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

Let me trace through this: if we have [1,3,5,7,9] and target=5, we start with left=0, right=4, mid=2. arr[2]=5 matches target, so we return 2."""),
    
    ("ai", """{{
    "session_id": "demo_session_123",
    "user_id": "demo_user_456", 
    "problem_solving_correctness": "Excellent approach. Candidate immediately recognized the optimal binary search solution and implemented it correctly with proper bounds handling.",
    "code_quality_maintainability": "Very good code quality. Clean, readable implementation with appropriate variable names. Could benefit from input validation comments.",
    "technical_knowledge_depth": "Strong technical foundation. Demonstrated understanding of time complexity, algorithm choice rationale, and provided clear trace-through example.", 
    "language_and_tool_proficiency": "Proficient in Python. Proper syntax, good use of language features, and clear coding style.",
    "problem_comprehension_and_clarification": "Excellent problem comprehension. Asked relevant clarifying questions about edge cases and requirements before coding.",
    "edge_case_handling_and_testing": "Good awareness of edge cases. Mentioned handling of target not found, though could have discussed empty array scenario.",
    "debugging_and_troubleshooting": "Demonstrated debugging mindset by walking through an example execution trace.",
    "hints_used_independence": "High independence. Required no hints and showed self-directed problem-solving approach.",
    "communication_reasoning": "Very clear communication. Explained approach, asked good questions, and provided step-by-step reasoning.",
    "receptiveness_to_feedback": "N/A - no feedback was given during this session.",
    "summary": "The candidate demonstrated excellent problem-solving skills and technical knowledge. They immediately identified the optimal solution, asked thoughtful clarifying questions, and implemented a clean, correct solution. Their communication was clear and they showed strong understanding of algorithms and complexity analysis. This is a strong performance that indicates solid software engineering fundamentals.",
    "problem_solving_correctness_score": 5,
    "code_quality_maintainability_score": 4, 
    "technical_knowledge_depth_score": 5,
    "overall_technical_score": 5
}}"""),
    
    MessagesPlaceholder(variable_name="messages"),
    
    ("human", "Based on the interview transcript above, provide a comprehensive evaluation following the same format and quality as the example. Be specific in your feedback and ensure your numeric scores align with your qualitative assessments.")
]).partial(format_instructions=parser.get_format_instructions())

# --- 3. Enhanced Main Function with Retry Logic ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
async def _invoke_evaluation_with_retry(chain, formatted_messages: List):
    """Invoke evaluation with retry logic for robustness."""
    try:
        response = await chain.ainvoke({"messages": formatted_messages})
        return response
    except Exception as e:
        logger.warning(f"Evaluation attempt failed: {e}")
        raise

async def generate_evaluation(session_id: str, user_id: str, messages: List[Dict]) -> Dict[str, Any]:
    """
    Runs the AI evaluation chain on a given interview transcript.
    
    Args:
        session_id: Unique session identifier
        user_id: Unique user identifier  
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        Dictionary containing the structured evaluation results
        
    Raises:
        ConnectionError: If LLM is not available or API call fails
        ValueError: If input data is invalid
        Exception: For other unexpected errors
    """
    
    # Validation
    if not llm:
        raise ConnectionError("AI evaluation service is not available. Please check API key configuration.")
    
    if not session_id or not session_id.strip():
        raise ValueError("Session ID cannot be empty")
        
    if not user_id or not user_id.strip():
        raise ValueError("User ID cannot be empty")
        
    if not messages:
        raise ValueError("Messages list cannot be empty")
        
    if len(messages) < 2:
        raise ValueError("Interview transcript must contain at least 2 messages")

    # Convert and validate messages
    formatted_messages = []
    try:
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")
                
            role = msg.get("role", "").lower().strip()
            content = msg.get("content", "").strip()
            
            if not role or not content:
                raise ValueError(f"Message {i} must have both 'role' and 'content' fields")
                
            if role not in ["ai", "human"]:
                raise ValueError(f"Message {i} role must be 'ai' or 'human', got: {role}")
            
            # Create appropriate message objects
            if role == "ai":
                formatted_messages.append(AIMessage(content=content))
            else:
                formatted_messages.append(HumanMessage(content=content))
                
    except Exception as e:
        logger.error(f"Error processing messages for session {session_id}: {e}")
        raise ValueError(f"Invalid message format: {str(e)}")

    # Validate message content quality
    total_content_length = sum(len(msg.content) for msg in formatted_messages)
    if total_content_length < 50:
        logger.warning(f"Very short transcript for session {session_id}")
        
    # Check for balanced conversation
    ai_messages = [msg for msg in formatted_messages if isinstance(msg, AIMessage)]
    human_messages = [msg for msg in formatted_messages if isinstance(msg, HumanMessage)]
    
    if len(ai_messages) == 0:
        raise ValueError("Transcript must contain at least one AI message")
    if len(human_messages) == 0:
        raise ValueError("Transcript must contain at least one human message")

    # Create evaluation chain
    chain = prompt | llm | parser

    try:
        logger.info(f"Starting evaluation for session: {session_id}, user: {user_id}")
        logger.debug(f"Processing {len(formatted_messages)} messages ({len(ai_messages)} AI, {len(human_messages)} human)")
        
        # Invoke with retry logic
        response_pydantic = await _invoke_evaluation_with_retry(chain, formatted_messages)
        
        # Convert to dictionary and add metadata
        evaluation_data = response_pydantic.model_dump()
        evaluation_data['session_id'] = session_id.strip()
        evaluation_data['user_id'] = user_id.strip()
        
        # Validate the response
        if not evaluation_data.get('summary'):
            logger.warning(f"No summary generated for session {session_id}")
            evaluation_data['summary'] = "Evaluation completed but summary not available."
        
        # Log success
        logger.info(f"Evaluation completed successfully for session: {session_id}")
        logger.debug(f"Generated evaluation with {len([k for k, v in evaluation_data.items() if v is not None])} populated fields")
        
        return evaluation_data
        
    except asyncio.TimeoutError:
        logger.error(f"Evaluation timeout for session {session_id}")
        raise ConnectionError("Evaluation request timed out. Please try again.")
    except Exception as e:
        logger.error(f"Evaluation failed for session {session_id}: {str(e)}")
        raise ConnectionError(f"AI evaluation service error: {str(e)}")

# --- 4. Utility Functions ---
def validate_api_key():
    """Validate that the API key is available and properly formatted."""
    if not google_api_key:
        return False, "GOOGLE_API_KEY not found in environment"
    if len(google_api_key) < 20:  # Basic length check
        return False, "GOOGLE_API_KEY appears to be invalid (too short)"
    return True, "API key validation passed"

async def test_llm_connection():
    """Test LLM connectivity with a simple request."""
    if not llm:
        return False, "LLM not initialized"
    
    try:
        test_messages = [HumanMessage(content="Hello, this is a connectivity test.")]
        test_chain = prompt | llm
        
        # Use a timeout for the test
        response = await asyncio.wait_for(
            test_chain.ainvoke({"messages": test_messages}),
            timeout=30
        )
        return True, "LLM connection test successful"
    except Exception as e:
        return False, f"LLM connection test failed: {str(e)}"

# Initialize and validate on module load
if __name__ == "__main__":
    # Run basic validation
    is_valid, message = validate_api_key()
    logger.info(f"API key validation: {message}")
    
    if is_valid and llm:
        logger.info("Evaluator module initialized successfully")
    else:
        logger.warning("Evaluator module initialized with warnings - check configuration")