"""
CodeSage: Complete Automated Technical Interview System
Built with LangGraph for agent orchestration
"""

import json
import time
from typing import TypedDict, List, Annotated, Literal, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client, Client

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

class Config:
    """Configuration constants for the CodeSage system"""
    GOOGLE_API_KEY = "AIzaSyCCwrq1xTZjuzSf29ql1-wpPvSyUjQiEIE"
    SUPABASE_URL = "https://nrckkprlzxhlzmgxppsa.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5yY2trcHJsenhobHptZ3hwcHNhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg5NTE0MjAsImV4cCI6MjA3NDUyNzQyMH0.BfniwBAsVzIvfY2Ddf-7v9iQsgyiW3zu7l_aImE9bHo"
    
    TIME_LIMITS = {
        "easy": 20 * 60,    # 20 minutes
        "medium": 30 * 60,  # 30 minutes
        "hard": 45 * 60     # 45 minutes
    }
    
    MAX_QUESTIONS = 3

# Initialize global services
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20", 
    google_api_key=Config.GOOGLE_API_KEY
)

supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """Central state for the interview system"""
    # Conversation history
    messages: Annotated[List[Dict[str, Any]], add_messages]
    
    # Current question tracking
    question_id: List[int]  # List to track question progression (now using int IDs)
    difficulty: List[str]   # Difficulty progression
    
    # Real-time candidate input
    code_script: str        # Current code in sandbox
    voice_transcript: str   # Speech-to-text transcript
    
    # Interview control
    start_time: float       # Interview start timestamp
    question_start_time: float  # Current question start time
    time_exceeded: bool     # Time limit flag
    interview_complete: bool # Interview completion flag
    
    # Performance tracking
    hints_used: int         # Number of hints provided
    performance_score: float # Running performance score

# =============================================================================
# DATABASE SERVICE LAYER
# =============================================================================

class DatabaseService:
    """Service layer for database operations"""
    
    @staticmethod
    def find_question_from_supabase(difficulty: str, topic: str) -> int:
        """
        Connects to Supabase and fetches a question ID based on difficulty and topic.
        """
        try:
            print(f"Searching for a '{difficulty}' question with topic '{topic}'...")
            
            # Build and execute the query using the Supabase client
            response = supabase.table('questions').select('id') \
                .eq('difficulty', difficulty) \
                .cs('topics', [topic]) \
                .execute()
                                
            # The data is in the 'data' attribute of the response
            if response.data:
                # Get the id from the first result in the list
                question_id = response.data[0]['id']
                print(f"Success! Found Question ID: {question_id}")
                return question_id
            else:
                print("No matching question found in Supabase.")
                return None
                
        except Exception as e:
            print(f"An error occurred while finding question: {e}")
            return None
    
    @staticmethod
    def get_question(question_id: int) -> Dict[str, Any]:
        """Fetch question details from database"""
        try:
            response = supabase.table('questions').select('*').eq('id', question_id).execute()
            if response.data:
                return response.data[0]
            return {"error": f"Question {question_id} not found"}
        except Exception as e:
            print(f"Database error fetching question: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def get_hints(question_id: int) -> List[Dict[str, Any]]:
        """Fetch available hints for a question"""
        try:
            response = supabase.table('hints').select('level', 'text').eq('question_id', question_id).execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"Database error fetching hints: {e}")
            return []
    
    @staticmethod
    def get_follow_ups(question_id: int) -> Dict[str, Any]:
        """Fetch follow-up questions for a question"""
        try:
            response = supabase.table('follow_ups').select('*').eq('question_id', question_id).execute()
            if response.data:
                return json.loads(response.data[0].get('follow_ups', '{}'))
            return {}
        except Exception as e:
            print(f"Database error fetching follow-ups: {e}")
            return {}
    
    @staticmethod
    def get_question_bank() -> List[Dict[str, Any]]:
        """Fetch all available questions"""
        try:
            response = supabase.table('questions').select('*').execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"Database error fetching question bank: {e}")
            # Fallback data for testing
            return [
                {
                    "id": 101,
                    "title": "Find Duplicates",
                    "difficulty": "easy",
                    "problem_statement": "Given an array of integers, find if the array contains any duplicates."
                },
                {
                    "id": 102, 
                    "title": "Two Sum",
                    "difficulty": "easy",
                    "problem_statement": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target."
                },
                {
                    "id": 201,
                    "title": "Longest Substring Without Repeating Characters",
                    "difficulty": "medium", 
                    "problem_statement": "Given a string s, find the length of the longest substring without repeating characters."
                },
                {
                    "id": 301,
                    "title": "Merge K Sorted Lists",
                    "difficulty": "hard",
                    "problem_statement": "You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it."
                }
            ]

# =============================================================================
# AGENT IMPLEMENTATIONS
# =============================================================================

def interviewer_agent(state: AgentState) -> Dict[str, Any]:
    """
    Central routing agent that analyzes the interview state and decides next action
    """
    print("🧠 INTERVIEWER AGENT: Analyzing current state...")
    
    # Get recent context
    recent_messages = state["messages"][-10:] if state["messages"] else []
    code_script = state.get("code_script", "")
    voice_transcript = state.get("voice_transcript", "")
    current_question = state["question_id"][-1] if state["question_id"] else "unknown"
    
    # Check if interview should end+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if len(state["question_id"]) >= Config.MAX_QUESTIONS:
        return {"next_agent": "eval_agent"}
    
    # Check time limits
    current_time = time.time()
    question_start = state.get("question_start_time", current_time)
    elapsed = current_time - question_start
    difficulty = state["difficulty"][-1] if state["difficulty"] else "medium"
    time_limit = Config.TIME_LIMITS.get(difficulty, Config.TIME_LIMITS["medium"])
    
    if elapsed >= time_limit:
        return {"next_agent": "question_setter_agent"}
    
    # Prepare prompt for LLM routing decision
    prompt = f"""You are an expert AI technical interviewer. Analyze the candidate's current state and decide the next action.

Current Situation:
- Question ID: {current_question}
- Code Snippet:
```python
{code_script}
```
- Voice Transcript: "{voice_transcript}"
- Recent Conversation: {recent_messages[-5:] if recent_messages else "No conversation yet"}
- Time Elapsed: {elapsed/60:.1f} minutes (Limit: {time_limit/60} minutes)

Decision Options:
- "HINT_AGENT": Candidate seems stuck, has bugs, or needs guidance
- "FOLLOW_UP_AGENT": Code looks good, probe deeper into understanding
- "QUESTION_SETTER_AGENT": Solution complete, move to next question
- "END_INTERVIEW": All questions completed

Respond with ONLY the agent name (e.g., "HINT_AGENT").
"""
    
    try:
        response = llm.invoke(prompt)
        decision = response.content.strip()
        print(f"🧠 INTERVIEWER AGENT: Routing to {decision}")
        
        # Map LLM response to actual agent names
        agent_mapping = {
            "HINT_AGENT": "hint_agent",
            "FOLLOW_UP_AGENT": "follow_up_agent", 
            "QUESTION_SETTER_AGENT": "question_setter_agent",
            "END_INTERVIEW": "eval_agent"
        }
        
        next_agent = agent_mapping.get(decision, "sandbox_analyzer")
        return {"next_agent": next_agent}
        
    except Exception as e:
        print(f"🧠 INTERVIEWER AGENT Error: {e}")
        return {"next_agent": "sandbox_analyzer"}

def hint_agent(state: AgentState) -> AgentState:
    """
    Provides contextual hints based on candidate's current progress
    """
    print("💡 HINT AGENT: Selecting appropriate hint...")
    
    current_question = state["question_id"][-1] if state["question_id"] else ""
    code_script = state.get("code_script", "")
    voice_transcript = state.get("voice_transcript", "")
    recent_messages = state["messages"][-5:] if state["messages"] else []
    
    # Get available hints from database
    hints = DatabaseService.get_hints(current_question)
    
    if not hints:
        # Fallback hints for testing
        hints = [
            {"level": "nudge", "text": "Think about how you can keep track of the numbers you've already seen."},
            {"level": "guide", "text": "Could a data structure with fast lookup times help you solve this efficiently?"},
            {"level": "direction", "text": "Try using a hash set to store elements as you iterate through the array."}
        ]
    
    prompt = f"""You are an expert technical interviewer providing hints to a candidate.

Candidate's Current State:
- Code: 
```python
{code_script}
```
- Transcript: "{voice_transcript}"
- Recent Discussion: {recent_messages}

Available Hints:
{json.dumps(hints, indent=2)}

Select the most appropriate hint based on their progress:
- "nudge" level: If they have some idea but are hesitant
- "guide" level: If they're on wrong track or using inefficient approach  
- "direction" level: If they're completely stuck or lost

Respond with ONLY the text of the chosen hint.
"""
    
    try:
        response = llm.invoke(prompt)
        chosen_hint = response.content.strip()
        
        print(f"💡 HINT AGENT: Providing hint: {chosen_hint[:50]}...")
        
        # Add hint to conversation
        state["messages"].append({
            "role": "assistant",
            "content": f"💡 **Hint:** {chosen_hint}",
            "timestamp": time.time()
        })
        
        # Track hint usage
        state["hints_used"] = state.get("hints_used", 0) + 1
        
        return state
        
    except Exception as e:
        print(f"💡 HINT AGENT Error: {e}")
        # Provide fallback hint
        state["messages"].append({
            "role": "assistant", 
            "content": "💡 **Hint:** Think about the most efficient way to approach this problem step by step.",
            "timestamp": time.time()
        })
        return state

def follow_up_agent(state: AgentState) -> AgentState:
    """
    Asks probing questions about time complexity, edge cases, optimizations
    """
    print("🔍 FOLLOW-UP AGENT: Analyzing solution for deeper questioning...")
    
    current_question = state["question_id"][-1] if state["question_id"] else ""
    code_script = state.get("code_script", "")
    voice_transcript = state.get("voice_transcript", "")
    
    # Get follow-up questions from database
    follow_ups = DatabaseService.get_follow_ups(current_question)
    
    if not follow_ups:
        # Fallback follow-ups for testing
        follow_ups = {
            "on_optimal_solution": [
                {"difficulty": "easy", "question": "Great! Can you explain the time complexity of your solution?"},
                {"difficulty": "medium", "question": "What about space complexity? Could we optimize it further?"}
            ],
            "on_suboptimal_solution": [
                {"difficulty": "easy", "question": "Your code works! Can you think of a more efficient approach?"},
                {"difficulty": "medium", "question": "What data structure might give us faster lookups?"}
            ]
        }
    
    prompt = f"""You are an expert technical interviewer. The candidate has written a solution. 
Analyze it and select the most appropriate follow-up question.

Candidate's Solution:
```python
{code_script}
```

Transcript: "{voice_transcript}"

Available Follow-up Questions:
{json.dumps(follow_ups, indent=2)}

Steps:
1. Determine if the solution is optimal (O(n) time) or suboptimal (O(n²) or worse)
2. Choose from "on_optimal_solution" or "on_suboptimal_solution" accordingly
3. Select the most appropriate question from that category

Respond with ONLY the question text.
"""
    try:
        response = llm.invoke(prompt)
        chosen_question = response.content.strip()
        
        print(f"🔍 FOLLOW-UP AGENT: Asking: {chosen_question[:50]}...")
        
        # Add follow-up question to conversation
        state["messages"].append({
            "role": "assistant",
            "content": f"🔍 {chosen_question}",
            "timestamp": time.time()
        })
        
        return state
        
    except Exception as e:
        print(f"🔍 FOLLOW-UP AGENT Error: {e}")
        # Provide fallback question
        state["messages"].append({
            "role": "assistant",
            "content": "🔍 Can you walk me through the time complexity of your solution?",
            "timestamp": time.time()
        })
        return state

def question_setter_agent(state: AgentState) -> AgentState:
    """
    Evaluates performance and selects the next question adaptively
    """
    print("📝 QUESTION SETTER AGENT: Selecting next question...")
    
    # Analyze performance on current question
    code_script = state.get("code_script", "")
    voice_transcript = state.get("voice_transcript", "")
    hints_used = state.get("hints_used", 0)
    recent_messages = state["messages"][-10:] if state["messages"] else []
    previous_question = state["question_id"][-1] if state["question_id"] else ""
    
    # Get question bank
    question_bank = DatabaseService.get_question_bank()
    
    # Filter out already asked questions
    asked_questions = set(state["question_id"])
    available_questions = [q for q in question_bank if q["id"] not in asked_questions]
    
    if not available_questions:
        print("📝 QUESTION SETTER: No more questions available")
        state["interview_complete"] = True
        return state
    
    prompt = f"""You are an expert technical interviewer selecting the next question based on candidate performance.

Previous Question Performance:
- Question ID: {previous_question}
- Final Code:
```python
{code_script}
```
- Voice Transcript: "{voice_transcript}"
- Hints Used: {hints_used}
- Conversation: {recent_messages[-3:] if recent_messages else []}

Performance Assessment Guidelines:
- Excellent: Optimal solution, few/no hints, clear explanation → harder question
- Good: Working solution, some hints needed → similar or slightly harder
- Struggling: Many hints, incomplete solution → easier question or different topic

Available Questions (JSON format):
{json.dumps(available_questions, indent=2)}

Respond with ONLY a JSON object for the selected question:
{{"id": "101", "problem_statement": "Given a string s, find..."}}
"""
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Parse the JSON response
        next_question = json.loads(response_text)
        question_id = int(next_question["id"])  # Convert to int
        problem_statement = next_question["problem_statement"]
        
        print(f"📝 QUESTION SETTER: Selected question {question_id}")
        
        # Update state for new question
        state["question_id"].append(question_id)
        
        # Get question details to determine difficulty
        question_details = DatabaseService.get_question(question_id)
        difficulty = question_details.get("difficulty", "medium")
        state["difficulty"].append(difficulty)
        
        # Reset for new question
        state["code_script"] = ""
        state["voice_transcript"] = ""
        state["hints_used"] = 0
        state["question_start_time"] = time.time()
        
        # Add new question to conversation
        state["messages"].append({
            "role": "assistant",
            "content": f"📝 **Next Question ({difficulty.upper()}):**\n\n{problem_statement}",
            "timestamp": time.time()
        })
        
        return state
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"📝 QUESTION SETTER Error: {e}")
        # End interview on error
        state["interview_complete"] = True
        state["messages"].append({
            "role": "assistant",
            "content": "Thank you for your time. That concludes our interview session.",
            "timestamp": time.time()
        })
        return state

def eval_agent(state: AgentState) -> AgentState:
    """
    Provides comprehensive final evaluation of the interview
    """
    print("📊 EVALUATION AGENT: Generating final assessment...")
    
    total_questions = len(state["question_id"])
    total_hints = state.get("hints_used", 0)
    difficulty_progression = state.get("difficulty", [])
    interview_duration = time.time() - state.get("start_time", time.time())
    
    # Collect performance data
    final_code = state.get("code_script", "")
    final_transcript = state.get("voice_transcript", "")
    conversation_history = state["messages"][-20:] if state["messages"] else []
    
    prompt = f"""You are an expert technical interviewer providing a comprehensive final evaluation.

Interview Summary:
- Total Questions Attempted: {total_questions}
- Difficulty Progression: {difficulty_progression}
- Total Hints Used: {total_hints}
- Interview Duration: {interview_duration/60:.1f} minutes
- Final Code Sample:
```python
{final_code}
```
- Final Transcript: "{final_transcript}"
- Key Conversation Moments: {conversation_history[-5:]}

Evaluation Criteria:
1. **Technical Skills** (40%): Code quality, algorithm efficiency, syntax correctness
2. **Problem-Solving** (30%): Logical approach, debugging ability, optimization thinking
3. **Communication** (20%): Clear explanation, think-aloud process, question handling  
4. **Adaptability** (10%): Response to hints, learning from feedback

Provide a structured evaluation with:
- Overall Score (1-10)
- Strengths (2-3 key points)
- Areas for Improvement (2-3 points)
- Specific Examples from the interview
- Final Recommendation (Strong Hire/Hire/No Hire/Strong No Hire)

Keep it professional, constructive, and specific.
"""
    
    try:
        response = llm.invoke(prompt)
        evaluation = response.content.strip()
        
        print("📊 EVALUATION AGENT: Assessment complete")
        
        # Add final evaluation
        state["messages"].append({
            "role": "assistant",
            "content": f"📊 **FINAL EVALUATION**\n\n{evaluation}\n\n---\nThank you for participating in this technical interview!",
            "timestamp": time.time()
        })
        
        state["interview_complete"] = True
        return state
        
    except Exception as e:
        print(f"📊 EVALUATION AGENT Error: {e}")
        state["messages"].append({
            "role": "assistant",
            "content": "📊 **Interview Complete** - Thank you for your time and effort!",
            "timestamp": time.time()
        })
        state["interview_complete"] = True
        return state

def initial_question_setter(difficulty: str, topic: str) -> int:
    """
    Sets up the initial question based on user preferences for difficulty and topic.
    """
    print(f"🎯 INITIAL QUESTION SETTER: Finding question for {difficulty} difficulty and {topic} topic...")
    
    # Use the database service to find matching question
    question_id = DatabaseService.find_question_from_supabase(difficulty, topic)
    
    if question_id:
        print(f"✅ Selected initial question ID: {question_id}")
        return question_id
    else:
        print("⚠️  No matching question found. Using fallback question.")
        # Return fallback question ID based on difficulty
        fallback_ids = {"easy": 101, "medium": 201, "hard": 301}
        return fallback_ids.get(difficulty, 101)

def sandbox_analyzer(state: AgentState) -> AgentState:
    """
    Processes real-time input from coding sandbox and voice transcript
    In production, this would integrate with actual sandbox and speech-to-text APIs
    """
    print("🔧 SANDBOX ANALYZER: Processing real-time input...")
    
    # In a real implementation, this would:
    # 1. Fetch latest code from the coding sandbox
    # 2. Get real-time voice transcript from speech-to-text service
    # 3. Update state with this fresh information
    
    # For demo purposes, we'll simulate some progression
    current_code = state.get("code_script", "")
    current_transcript = state.get("voice_transcript", "")
    
    # Simulate code progression if empty
    if not current_code:
        state["code_script"] = "# Let me think about this problem...\n# I need to find duplicates in an array"
    
    # Simulate voice transcript if empty
    if not current_transcript:
        state["voice_transcript"] = "Let me understand the problem first. I need to check for duplicates..."
    
    return state
    """
    Processes real-time input from coding sandbox and voice transcript
    In production, this would integrate with actual sandbox and speech-to-text APIs
    """
    print("🔧 SANDBOX ANALYZER: Processing real-time input...")
    
    # In a real implementation, this would:
    # 1. Fetch latest code from the coding sandbox
    # 2. Get real-time voice transcript from speech-to-text service
    # 3. Update state with this fresh information
    
    # For demo purposes, we'll simulate some progression
    current_code = state.get("code_script", "")
    current_transcript = state.get("voice_transcript", "")
    
    # Simulate code progression if empty
    if not current_code:
        state["code_script"] = "# Let me think about this problem...\n# I need to find duplicates in an array"
    
    # Simulate voice transcript if empty
    if not current_transcript:
        state["voice_transcript"] = "Let me understand the problem first. I need to check for duplicates..."
    
    return state

# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_interview_graph():
    """
    Constructs the complete interview workflow graph
    """
    print("🚀 Building CodeSage Interview Graph...")
    
    # Initialize graph with state
    workflow = StateGraph(AgentState)
    
    # Add all agent nodes
    workflow.add_node("interviewer_agent", interviewer_agent)
    workflow.add_node("hint_agent", hint_agent)
    workflow.add_node("follow_up_agent", follow_up_agent) 
    workflow.add_node("question_setter_agent", question_setter_agent)
    workflow.add_node("eval_agent", eval_agent)
    workflow.add_node("sandbox_analyzer", sandbox_analyzer)
    
    # Define routing functions
    def route_interviewer(state: AgentState) -> str:
        """Route from interviewer based on its decision"""
        # The interviewer_agent returns a dict with next_agent
        # This is handled by the conditional edge logic
        return "sandbox_analyzer"  # Default fallback
    
    def route_after_agents(state: AgentState) -> str:
        """Route back to interviewer after other agents"""
        if state.get("interview_complete", False):
            return END
        return "interviewer_agent"
    
    def route_from_eval(state: AgentState) -> str:
        """End interview after evaluation"""
        return END
    
    # Set entry point
    workflow.set_entry_point("sandbox_analyzer")
    
    # Add edges
    workflow.add_edge("sandbox_analyzer", "interviewer_agent")
    
    # Conditional routing from interviewer
    workflow.add_conditional_edges(
        "interviewer_agent",
        lambda state: interviewer_agent(state)["next_agent"],
        {
            "hint_agent": "hint_agent",
            "follow_up_agent": "follow_up_agent",
            "question_setter_agent": "question_setter_agent", 
            "eval_agent": "eval_agent",
            "sandbox_analyzer": "sandbox_analyzer"
        }
    )
    
    # Route back to analysis after actions
    workflow.add_edge("hint_agent", "sandbox_analyzer")
    workflow.add_edge("follow_up_agent", "sandbox_analyzer")
    workflow.add_edge("question_setter_agent", "sandbox_analyzer")
    
    # Evaluation ends the interview
    workflow.add_edge("eval_agent", END)
    
    return workflow.compile()

# =============================================================================
# MAIN INTERFACE
# =============================================================================

class CodeSageInterview:
    """Main interface for running CodeSage technical interviews"""
    
    def __init__(self):
        self.graph = create_interview_graph()
    
    def start_interview(self, candidate_name: str = "Candidate", initial_difficulty: str = "easy", topic: str = "Array") -> Dict[str, Any]:
        """
        Start a new technical interview session
        
        Args:
            candidate_name: Name of the candidate
            initial_difficulty: Starting difficulty level
            topic: Topic preference for first question
            
        Returns:
            Complete interview results
        """
        print("=" * 60)
        print("🎯 CODESAGE TECHNICAL INTERVIEW SYSTEM")
        print("=" * 60)
        print(f"👤 Candidate: {candidate_name}")
        print(f"⚡ Starting Difficulty: {initial_difficulty.upper()}")
        print(f"📚 Preferred Topic: {topic}")
        print(f"🕐 Session Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Get initial question based on user preferences
        initial_question_id = initial_question_setter(initial_difficulty, topic)
        
        # Get the question details to show to candidate
        question_details = DatabaseService.get_question(initial_question_id)
        initial_problem = question_details.get("problem_statement", "Loading problem...")
        
        # Initialize interview state
        start_time = time.time()
        
        initial_state = AgentState(
            messages=[{
                "role": "assistant",
                "content": f"Welcome {candidate_name}! I'm your AI technical interviewer. We'll work through some coding problems together.\n\n📝 **First Question ({initial_difficulty.upper()}):**\n\n{initial_problem}",
                "timestamp": start_time
            }],
            question_id=[initial_question_id],  # Start with selected question
            difficulty=[initial_difficulty],
            code_script="",
            voice_transcript="",
            start_time=start_time,
            question_start_time=start_time,
            time_exceeded=False,
            interview_complete=False,
            hints_used=0,
            performance_score=0.0
        )
        
        try:
            # Run the interview graph
            final_state = self.graph.invoke(initial_state)
            
            # Process results
            results = self._process_results(final_state, candidate_name)
            
            print("\n" + "=" * 60)
            print("✅ INTERVIEW SESSION COMPLETED")
            print("=" * 60)
            
            return results
            
        except Exception as e:
            print(f"\n❌ Interview Error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "candidate": candidate_name
            }
    
    def _process_results(self, final_state: AgentState, candidate_name: str) -> Dict[str, Any]:
        """Process and format interview results"""
        
        total_duration = time.time() - final_state.get("start_time", time.time())
        
        results = {
            "status": "completed",
            "candidate_name": candidate_name,
            "interview_duration_minutes": round(total_duration / 60, 2),
            "questions_attempted": len(final_state.get("question_id", [])),
            "difficulty_progression": final_state.get("difficulty", []),
            "total_hints_used": final_state.get("hints_used", 0),
            "conversation_length": len(final_state.get("messages", [])),
            "final_code": final_state.get("code_script", ""),
            "final_transcript": final_state.get("voice_transcript", ""),
            "complete_conversation": final_state.get("messages", [])
        }
        
        # Extract evaluation from last message if available
        messages = final_state.get("messages", [])
        if messages:
            last_message = messages[-1].get("content", "")
            if "FINAL EVALUATION" in last_message:
                results["evaluation"] = last_message
        
        return results
    
    def print_conversation(self, results: Dict[str, Any]):
        """Print the complete interview conversation"""
        if "complete_conversation" not in results:
            print("No conversation data available")
            return
        
        print("\n" + "=" * 60)
        print("💬 COMPLETE INTERVIEW CONVERSATION")
        print("=" * 60)
        
        for i, message in enumerate(results["complete_conversation"]):
            role_icon = "🤖" if message["role"] == "assistant" else "👤"
            timestamp = message.get("timestamp", time.time())
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            print(f"\n[{time_str}] {role_icon} {message['role'].upper()}:")
            print("-" * 40)
            print(message["content"])

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

# def main():
#     """Example usage of the CodeSage interview system"""
    
#     # Create interview instance
#     interview_system = CodeSageInterview()
    
#     # Start interview
#     results = interview_system.start_interview(
#         candidate_name="Alice Johnson",
#         initial_difficulty="easy",
#         topic="Array"
#     )
    
#     # Print summary
#     if results["status"] == "completed":
#         print(f"\n📋 Interview Summary for {results['candidate_name']}:")
#         print(f"⏱️  Duration: {results['interview_duration_minutes']} minutes")
#         print(f"❓ Questions: {results['questions_attempted']}")
#         print(f"📈 Difficulty: {' → '.join(results['difficulty_progression'])}")
#         print(f"💡 Hints Used: {results['total_hints_used']}")
        
#         # Print conversation
#         interview_system.print_conversation(results)
    
#     return results

# if __name__ == "__main__":
#     # Run the example
#     results = main()