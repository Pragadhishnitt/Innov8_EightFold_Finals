import asyncio
import json
import logging
import base64
import time
import hashlib
import httpx
from typing import Dict, List, Optional, Any
from fastapi import Response
import asyncio

from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import uvicorn
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle missing modules with stubs
try:
    from database_manager import db_manager
except ImportError:
    logger.warning("database_manager not found, using stub")
    class StubDBManager:
        async def store_session(self, session_id): pass
        async def update_session_activity(self, session_id, active): pass
        async def store_code_update(self, session_id, code_entry): pass
        async def store_execution_result(self, session_id, result_entry): pass
        async def store_audio_chunk(self, session_id, audio_entry): return "stub_chunk_id"
        async def get_audio_chunk(self, session_id, chunk_index): return None
        async def get_audio_chunks_metadata(self, session_id): return []
        async def get_session_transcripts(self, session_id): return []
        async def get_full_session_transcript(self, session_id): return ""
        async def test_connection(self): return "stub connection"
    db_manager = StubDBManager()

try:
    from audio_processor import process_audio_chunk_from_db, process_pending_audio_chunks
except ImportError:
    logger.warning("audio_processor not found, using stubs")
    async def process_audio_chunk_from_db(chunk_id): return True
    async def process_pending_audio_chunks(): pass

try:
    from transcript_service import transcript_service
except ImportError:
    logger.warning("transcript_service not found, using stub")
    class StubTranscriptService:
        async def generate_transcript_for_chunk(self, chunk_id, session_id): 
            return {"success": True, "transcript": "stub transcript"}
        async def get_transcript_file(self, session_id, chunk_id): return None
        async def list_session_transcripts(self, session_id): return []
    transcript_service = StubTranscriptService()

try:
    from snapshot_service import snapshot_service
except ImportError:
    logger.warning("snapshot_service not found, using stub")
    class StubSnapshotService:
        async def get_latest_snapshot(self): return None
        async def get_snapshot_by_id(self, snapshot_id): return None
        async def list_snapshots(self): return []
        async def take_snapshot(self): return {"success": True, "snapshot_id": "stub"}
    snapshot_service = StubSnapshotService()

# Security and validation models
security = HTTPBearer(auto_error=False)

# Judge0 Configuration
JUDGE0_URL = "https://ce.judge0.com"
JUDGE0_HEADERS = {"Content-Type": "application/json"}

# Supported languages with Judge0 IDs
SUPPORTED_LANGUAGES = {
    "python": 71,
    "javascript": 63,
    "c": 50,
    "cpp": 54,
    "java": 62,
    "go": 60,
    "rust": 73,
    "typescript": 74
}

class CodeExecutionRequest(BaseModel):
    code: str = Field(..., max_length=50000)
    language: str = Field(default="python", pattern="^(python|javascript|java|go|c|cpp|rust|typescript)$")
    timeout: int = Field(default=5, ge=1, le=30)
    stdin: str = Field(default="", max_length=10000)
    test_cases: Optional[List[Dict[str, str]]] = Field(default=None)
    estimate_big_o: bool = Field(default=False)
    
    @field_validator('code')
    @classmethod
    def validate_code_safety(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Code cannot be empty')
        return v

class TestCase(BaseModel):
    stdin: str = Field(..., max_length=10000)
    expected_output: Optional[str] = Field(default=None, max_length=10000)
    description: Optional[str] = Field(default="", max_length=200)

class CodeAnalysisRequest(BaseModel):
    code: str = Field(..., max_length=50000)
    language: str = Field(default="python")
    test_cases: List[TestCase] = Field(default=[])
    estimate_big_o: bool = Field(default=False)

class AudioChunkRequest(BaseModel):
    audio_data: str = Field(..., max_length=1000000)
    format: str = Field(default="webm", pattern="^(webm|mp3|wav|ogg)$")
    duration: float = Field(default=0, ge=0, le=300)

class CodeUpdateRequest(BaseModel):
    code: str = Field(..., max_length=50000)
    cursor_position: int = Field(default=0, ge=0)

# Rate limiting
class RateLimiter:
    def __init__(self, max_requests: int = 10, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        now = time.time()
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window
        ]
        
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        return False

# Enhanced Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_data: Dict[str, Dict] = {}
        self.rate_limiter = RateLimiter(max_requests=30, window=60)
        self.max_sessions = 100
        self.session_ttl = timedelta(hours=2)
        
    async def connect(self, websocket: WebSocket, session_id: str) -> bool:
        if len(self.active_connections) >= self.max_sessions:
            await websocket.close(code=1008, reason="Server at capacity")
            return False
            
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_data[session_id] = {
            "code_history": [],
            "audio_chunks": [],
            "execution_results": [],
            "connected_at": datetime.now(),
            "last_activity": datetime.now(),
            "execution_count": 0,
            "total_audio_duration": 0,
            "resource_usage": {
                "cpu_time": 0,
                "memory_peak": 0
            }
        }
        logger.info(f"Client connected: {session_id}")
        await db_manager.store_session(session_id)
        return True

    async def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_data:
            self.session_data[session_id]["disconnected_at"] = datetime.now()
        logger.info(f"Client disconnected: {session_id}")
        await db_manager.update_session_activity(session_id, False)

    async def send_personal_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                await self.disconnect(session_id)

    async def broadcast(self, message: dict):
        disconnected = []
        for session_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to broadcast to {session_id}: {e}")
                disconnected.append(session_id)
        
        for session_id in disconnected:
            await self.disconnect(session_id)
    
    async def cleanup_old_sessions(self):
        cutoff_time = datetime.now() - self.session_ttl
        to_remove = []
        
        for session_id, data in self.session_data.items():
            if (session_id not in self.active_connections and 
                data.get("last_activity", datetime.min) < cutoff_time):
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.session_data[session_id]
            logger.info(f"Cleaned up old session: {session_id}")

# Judge0 Code Execution Service
class Judge0Service:
    def __init__(self):
        self.base_url = JUDGE0_URL
        self.headers = JUDGE0_HEADERS
        self.rate_limiter = RateLimiter(max_requests=20, window=60)
    
    async def submit_code(self, source_code: str, language_id: int, stdin: str = "", timeout: int = 5) -> Dict[str, Any]:
        if not self.rate_limiter.is_allowed("judge0_api"):
            raise Exception("Rate limit exceeded for code execution")
        
        submission_data = {
            "source_code": source_code,
            "language_id": language_id,
            "stdin": stdin,
            "cpu_time_limit": min(timeout, 10),
            "memory_limit": 256000,
            "wall_time_limit": min(timeout + 5, 15),
            "enable_per_process_and_thread_time_limit": True,
            "enable_per_process_and_thread_memory_limit": True
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/submissions?base64_encoded=false&wait=true",
                    headers=self.headers,
                    json=submission_data
                )
                
                if response.status_code in [200, 201]:
                    return response.json()
                else:
                    raise Exception(f"Judge0 submission failed: {response.status_code} - {response.text}")
                    
            except httpx.TimeoutException:
                raise Exception("Judge0 request timeout")
            except Exception as e:
                raise Exception(f"Judge0 request failed: {str(e)}")
    
    async def execute_code(self, code: str, language: str, stdin: str = "", timeout: int = 5) -> Dict[str, Any]:
        if language not in SUPPORTED_LANGUAGES:
            return {
                "success": False,
                "output": "",
                "error": f"Unsupported language: {language}",
                "execution_time": "0ms",
                "memory_usage": "0KB",
                "security_level": "Judge0 Sandboxed",
                "status": "Language Error"
            }
        
        language_id = SUPPORTED_LANGUAGES[language]
        
        try:
            result = await self.submit_code(code, language_id, stdin, timeout)
            
            status_desc = result.get("status", {}).get("description", "Unknown")
            stdout = result.get("stdout", "") or ""
            stderr = result.get("stderr", "") or ""
            compile_output = result.get("compile_output", "") or ""
            time_taken = result.get("time", "0")
            memory_used = result.get("memory", "0")
            
            success = status_desc == "Accepted"
            output = stdout.strip() if success else ""
            error_parts = []
            
            if compile_output:
                error_parts.append(f"Compile Error: {compile_output}")
            if stderr:
                error_parts.append(f"Runtime Error: {stderr}")
            if not success and status_desc != "Accepted":
                error_parts.append(f"Status: {status_desc}")
            
            error = "\n".join(error_parts) if error_parts else ""
            
            return {
                "success": success,
                "output": output,
                "error": error,
                "execution_time": f"{float(time_taken or 0) * 1000:.0f}ms" if time_taken else "0ms",
                "memory_usage": f"{int(memory_used or 0)}KB" if memory_used else "0KB",
                "security_level": "Judge0 Sandboxed",
                "status": status_desc,
                "raw_result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time": "N/A",
                "memory_usage": "N/A",
                "security_level": "Judge0 Sandboxed",
                "status": "Execution Failed"
            }
    
    async def run_test_cases(self, code: str, language: str, test_cases: List[Dict], timeout: int = 5) -> Dict[str, Any]:
        results = []
        passed_count = 0
        total_time = 0
        max_memory = 0
        
        for i, test_case in enumerate(test_cases):
            stdin = test_case.get("stdin", "")
            expected = test_case.get("expected_output")
            description = test_case.get("description", f"Test case {i+1}")
            
            try:
                result = await self.execute_code(code, language, stdin, timeout)
                
                is_correct = True
                if expected is not None:
                    actual_output = result.get("output", "").strip()
                    expected_output = expected.strip()
                    is_correct = actual_output == expected_output
                    
                    if not is_correct:
                        result["error"] = result.get("error", "") + f"\nExpected: '{expected_output}'\nActual: '{actual_output}'"
                
                if result["success"] and is_correct:
                    passed_count += 1
                
                time_str = result.get("execution_time", "0ms").replace("ms", "")
                try:
                    total_time += float(time_str)
                except:
                    pass
                
                memory_str = result.get("memory_usage", "0KB").replace("KB", "")
                try:
                    max_memory = max(max_memory, int(memory_str))
                except:
                    pass
                
                results.append({
                    "test_case": i + 1,
                    "description": description,
                    "stdin": stdin,
                    "expected_output": expected,
                    "actual_output": result.get("output", ""),
                    "success": result["success"] and is_correct,
                    "error": result.get("error", ""),
                    "execution_time": result.get("execution_time", "N/A"),
                    "memory_usage": result.get("memory_usage", "N/A"),
                    "status": result.get("status", "Unknown")
                })
                
            except Exception as e:
                results.append({
                    "test_case": i + 1,
                    "description": description,
                    "stdin": stdin,
                    "expected_output": expected,
                    "actual_output": "",
                    "success": False,
                    "error": str(e),
                    "execution_time": "N/A",
                    "memory_usage": "N/A",
                    "status": "Test Failed"
                })
        
        return {
            "results": results,
            "summary": {
                "total_tests": len(test_cases),
                "passed": passed_count,
                "failed": len(test_cases) - passed_count,
                "success_rate": f"{(passed_count / len(test_cases) * 100):.1f}%" if test_cases else "0%",
                "total_execution_time": f"{total_time:.0f}ms",
                "max_memory_usage": f"{max_memory}KB"
            }
        }

# Enhanced Code Analysis
class EnhancedCodeAnalyzer:
    @staticmethod
    def analyze_code(code: str, language: str = "python") -> dict:
        analysis = {
            "syntax_valid": True,
            "complexity": "Low",
            "style_issues": [],
            "potential_errors": [],
            "security_issues": [],
            "performance_hints": [],
            "code_quality_score": 100,
            "language": language,
            "line_count": len(code.split('\n')),
            "char_count": len(code),
            "estimated_time_complexity": "O(1)"
        }
        
        if language == "python":
            return EnhancedCodeAnalyzer._analyze_python(code, analysis)
        elif language in ["javascript", "typescript"]:
            return EnhancedCodeAnalyzer._analyze_javascript(code, analysis)
        elif language in ["java"]:
            return EnhancedCodeAnalyzer._analyze_java(code, analysis)
        elif language in ["c", "cpp"]:
            return EnhancedCodeAnalyzer._analyze_c_cpp(code, analysis)
        elif language == "go":
            return EnhancedCodeAnalyzer._analyze_go(code, analysis)
        else:
            return EnhancedCodeAnalyzer._analyze_generic(code, analysis)
    
    @staticmethod
    def _analyze_python(code: str, analysis: dict) -> dict:
        try:
            compile(code, '<string>', 'exec')
            analysis["syntax_valid"] = True
        except SyntaxError as e:
            analysis["syntax_valid"] = False
            analysis["potential_errors"].append(f"Syntax Error: {str(e)}")
            analysis["code_quality_score"] -= 30
        
        nested_loops = code.count('for ') + code.count('while ')
        if nested_loops >= 3:
            analysis["complexity"] = "High"
            analysis["estimated_time_complexity"] = "O(n³) or higher"
        elif nested_loops == 2:
            analysis["complexity"] = "Medium" 
            analysis["estimated_time_complexity"] = "O(n²)"
        elif nested_loops == 1:
            analysis["estimated_time_complexity"] = "O(n)"
        
        dangerous_patterns = ['eval(', 'exec(', 'import os', 'import subprocess']
        for pattern in dangerous_patterns:
            if pattern in code:
                analysis["security_issues"].append(f"Potentially dangerous: {pattern}")
                analysis["code_quality_score"] -= 15
        
        return analysis
    
    @staticmethod
    def _analyze_javascript(code: str, analysis: dict) -> dict:
        if 'eval(' in code:
            analysis["security_issues"].append("Dangerous eval() usage")
            analysis["code_quality_score"] -= 20
        
        loop_count = code.count('for(') + code.count('for (') + code.count('while(') + code.count('while (')
        if loop_count >= 2:
            analysis["complexity"] = "Medium"
            analysis["estimated_time_complexity"] = "O(n²)"
        elif loop_count == 1:
            analysis["estimated_time_complexity"] = "O(n)"
        
        return analysis
    
    @staticmethod 
    def _analyze_java(code: str, analysis: dict) -> dict:
        if 'System.exit(' in code:
            analysis["security_issues"].append("System.exit() usage")
        
        loop_patterns = ['for(', 'for (', 'while(', 'while (', 'do {']
        loop_count = sum(code.count(pattern) for pattern in loop_patterns)
        
        if loop_count >= 2:
            analysis["complexity"] = "Medium"
            analysis["estimated_time_complexity"] = "O(n²)"
        elif loop_count == 1:
            analysis["estimated_time_complexity"] = "O(n)"
            
        return analysis
    
    @staticmethod
    def _analyze_c_cpp(code: str, analysis: dict) -> dict:
        dangerous_functions = ['system(', 'exec(', 'scanf(', 'gets(']
        for func in dangerous_functions:
            if func in code:
                analysis["security_issues"].append(f"Potentially unsafe function: {func}")
                analysis["code_quality_score"] -= 10
        
        if 'malloc(' in code and 'free(' not in code:
            analysis["potential_errors"].append("Possible memory leak - malloc without free")
            
        return analysis
    
    @staticmethod
    def _analyze_go(code: str, analysis: dict) -> dict:
        if 'os/exec' in code:
            analysis["security_issues"].append("Command execution detected")
            
        if 'go func(' in code or 'go ' in code:
            analysis["complexity"] = "Medium"
            analysis["performance_hints"].append("Concurrent code - ensure proper synchronization")
            
        return analysis
    
    @staticmethod
    def _analyze_generic(code: str, analysis: dict) -> dict:
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if len(non_empty_lines) > 50:
            analysis["complexity"] = "High"
        elif len(non_empty_lines) > 20:
            analysis["complexity"] = "Medium"
            
        return analysis

def detect_language_from_code(code: str) -> str:
    code_lower = code.lower().strip()
    
    if any(keyword in code_lower for keyword in ['def ', 'import ', 'print(', 'len(', 'range(']):
        return "python"
    elif any(keyword in code_lower for keyword in ['function ', 'const ', 'let ', 'var ', 'console.log']):
        return "javascript"
    elif any(keyword in code_lower for keyword in ['public class', 'system.out.print', 'string[]']):
        return "java"
    elif any(keyword in code_lower for keyword in ['#include', 'int main(', 'printf(', 'cout <<']):
        return "cpp" if 'cout' in code_lower or 'cin' in code_lower else "c"
    elif any(keyword in code_lower for keyword in ['func main(', 'package main', 'fmt.print']):
        return "go"
    elif any(keyword in code_lower for keyword in ['fn main(', 'println!', 'let mut']):
        return "rust"
    else:
        return "python"

def generate_enhanced_suggestions(code: str, analysis: dict, language: str) -> List[str]:
    suggestions = []
    
    if not analysis["syntax_valid"]:
        suggestions.append(f"Fix syntax errors in your {language} code")
    
    if analysis["complexity"] == "High":
        suggestions.append("Consider optimizing your algorithm - current complexity seems high")
    elif analysis["complexity"] == "Medium":
        suggestions.append("Good structure! Consider if nested loops can be optimized")
    
    if analysis["security_issues"]:
        suggestions.append("Address security concerns - avoid dangerous functions")
    
    if analysis["code_quality_score"] > 80:
        suggestions.append("Excellent code quality!")
    elif analysis["code_quality_score"] > 60:
        suggestions.append("Good progress! Address style issues for cleaner code")
    else:
        suggestions.append("Focus on fixing the highlighted issues")
    
    if language == "python" and "import" not in code and len(code) > 100:
        suggestions.append("Consider using appropriate Python libraries for your solution")
    elif language == "java" and "public class" not in code:
        suggestions.append("Remember to wrap your code in a proper Java class structure")
    elif language in ["c", "cpp"] and "#include" not in code:
        suggestions.append("Don't forget to include necessary header files")
    
    return suggestions

# Initialize services
manager = ConnectionManager()
judge0_service = Judge0Service()
code_analyzer = EnhancedCodeAnalyzer()

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting background tasks...")
    cleanup_task = asyncio.create_task(cleanup_sessions_periodically())
    audio_processing_task = asyncio.create_task(background_audio_processing())
    
    yield
    
    # Shutdown
    logger.info("Shutting down background tasks...")
    cleanup_task.cancel()
    audio_processing_task.cancel()
    try:
        await cleanup_task
        await audio_processing_task
    except asyncio.CancelledError:
        pass

async def cleanup_sessions_periodically():
    """Background task to clean up old sessions"""
    while True:
        try:
            await manager.cleanup_old_sessions()
            await asyncio.sleep(300)  # Clean up every 5 minutes
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
            await asyncio.sleep(60)

async def background_audio_processing():
    """Background task to process pending audio chunks"""
    while True:
        try:
            await process_pending_audio_chunks()
            await asyncio.sleep(30)  # Check every 30 seconds
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Background audio processing error: {e}")
            await asyncio.sleep(60)

# Initialize FastAPI app
app = FastAPI(
    title="CodeSage AI Interviewer",
    description="Enhanced AI-powered coding interview platform with Judge0 integration",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Static files - only mount if directory exists
import os
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    logger.warning("Static directory not found - creating it")
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Enhanced WebSocket message handlers
async def handle_code_execution(session_id: str, message: dict):
    try:
        session_data = manager.session_data[session_id]
        if session_data["execution_count"] >= 20:
            await manager.send_personal_message({
                "type": "error", 
                "message": "Execution limit reached for this session (20 executions max)"
            }, session_id)
            return
        
        try:
            exec_data = CodeExecutionRequest(**message)
        except Exception as e:
            await manager.send_personal_message({
                "type": "error", 
                "message": f"Invalid execution request: {str(e)}"
            }, session_id)
            return
        
        code = exec_data.code
        language = exec_data.language
        timeout = exec_data.timeout
        stdin = exec_data.stdin
        test_cases = exec_data.test_cases
        
        if test_cases:
            execution_result = await judge0_service.run_test_cases(
                code, language, test_cases, timeout
            )
            result_type = "test_results"
        else:
            execution_result = await judge0_service.execute_code(
                code, language, stdin, timeout
            )
            result_type = "execution_result"
        
        result_entry = {
            "code": code[:500],
            "code_hash": hashlib.md5(code.encode()).hexdigest(),
            "result": execution_result,
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "result_type": result_type
        }
        
        session_data["execution_results"].append(result_entry)
        await db_manager.store_execution_result(session_id, result_entry)
        session_data["execution_count"] += 1
        
        response = {
            "type": result_type,
            "timestamp": result_entry["timestamp"],
            "result": execution_result,
            "session_stats": {
                "total_executions": session_data["execution_count"],
                "executions_remaining": 20 - session_data["execution_count"]
            }
        }
        
        await manager.send_personal_message(response, session_id)
        
    except Exception as e:
        logger.error(f"Code execution error for session {session_id}: {e}")
        await manager.send_personal_message({
            "type": "error", 
            "message": f"Execution failed: {str(e)}"
        }, session_id)

async def handle_code_update(session_id: str, message: dict):
    try:
        update_data = CodeUpdateRequest(**message)
        
        code = update_data.code
        cursor_position = update_data.cursor_position
        timestamp = datetime.now()
        
        language = detect_language_from_code(code)
        
        code_entry = {
            "code": code,
            "cursor_position": cursor_position,
            "timestamp": timestamp.isoformat(),
            "line_count": len(code.split('\n')),
            "char_count": len(code),
            "language": language,
            "hash": hashlib.md5(code.encode()).hexdigest()
        }
        
        session_data = manager.session_data[session_id]
        session_data["code_history"].append(code_entry)
        await db_manager.store_code_update(session_id, code_entry)
        
        if len(session_data["code_history"]) > 50:
            session_data["code_history"] = session_data["code_history"][-50:]
        
        analysis = code_analyzer.analyze_code(code, language)
        suggestions = generate_enhanced_suggestions(code, analysis, language)
        
        response = {
            "type": "code_analysis",
            "timestamp": timestamp.isoformat(),
            "analysis": analysis,
            "suggestions": suggestions,
            "session_stats": {
                "total_updates": len(session_data["code_history"]),
                "current_quality_score": analysis["code_quality_score"],
                "detected_language": language
            }
        }
        
        await manager.send_personal_message(response, session_id)
        
    except Exception as e:
        logger.error(f"Code update error for session {session_id}: {e}")
        await manager.send_personal_message(
            {"type": "error", "message": "Failed to process code update"}, 
            session_id
        )

async def handle_audio_chunk(session_id: str, message: dict):
    try:
        logger.info(f"Processing audio chunk for session {session_id}")
        
        if 'audio_data' not in message:
            logger.error("No 'audio_data' key in message!")
            await manager.send_personal_message({
                "type": "error",
                "message": "No audio_data field in message"
            }, session_id)
            return
        
        audio_data_raw = message.get('audio_data')
        audio_format = message.get("format", "webm")
        duration = float(message.get("duration", 0))
        
        if isinstance(audio_data_raw, str) and audio_data_raw.startswith('data:'):
            try:
                _, audio_data_b64 = audio_data_raw.split(',', 1)
            except ValueError:
                logger.error("Invalid data URL format")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid data URL format"
                }, session_id)
                return
        else:
            audio_data_b64 = str(audio_data_raw)
        
        if not audio_data_b64 or len(audio_data_b64) < 100:
            await manager.send_personal_message({
                "type": "error",
                "message": f"Audio data too short: {len(audio_data_b64)} characters"
            }, session_id)
            return
        
        try:
            decoded = base64.b64decode(audio_data_b64)
        except Exception as e:
            await manager.send_personal_message({
                "type": "error",
                "message": f"Invalid base64 audio data: {str(e)}"
            }, session_id)
            return
        
        session_data = manager.session_data.get(session_id)
        if not session_data:
            await manager.send_personal_message({
                "type": "error",
                "message": "Session not found"
            }, session_id)
            return
        
        total_duration = session_data.get("total_audio_duration", 0)
        if total_duration + duration > 1800:
            await manager.send_personal_message({
                "type": "error",
                "message": f"Audio duration limit exceeded (current: {total_duration}s)"
            }, session_id)
            return
        
        timestamp = datetime.now()
        audio_entry = {
            "format": audio_format,
            "timestamp": timestamp.isoformat(),
            "duration": duration,
            "size_bytes": len(audio_data_b64),
            "audio_hash": hashlib.md5(audio_data_b64.encode()).hexdigest(),
            "audio_data": audio_data_b64,
            "chunk_index": len(session_data.get("audio_chunks", []))
        }
        
        try:
            chunk_id = await db_manager.store_audio_chunk(session_id, audio_entry)
            if chunk_id is None:
                raise Exception("Database returned None for chunk_id")
        except Exception as db_error:
            await manager.send_personal_message({
                "type": "error",
                "message": f"Database storage failed: {str(db_error)}"
            }, session_id)
            return
        
        session_data["audio_chunks"].append({
            "chunk_id": chunk_id,
            "timestamp": timestamp.isoformat(),
            "duration": duration,
            "format": audio_format
        })
        session_data["total_audio_duration"] = total_duration + duration
        
        asyncio.create_task(process_audio_chunk_from_db(str(chunk_id)))
        
        await manager.send_personal_message({
            "type": "audio_received",
            "timestamp": timestamp.isoformat(),
            "chunk_id": chunk_id,
            "message": "Audio chunk stored and processing started",
            "session_stats": {
                "total_audio_chunks": len(session_data["audio_chunks"]),
                "total_duration": session_data["total_audio_duration"],
                "remaining_duration": 1800 - session_data["total_audio_duration"]
            }
        }, session_id)
        
    except Exception as e:
        logger.error(f"Critical error in handle_audio_chunk: {e}", exc_info=True)
        await manager.send_personal_message({
            "type": "error",
            "message": f"Failed to process audio: {str(e)}",
            "error_type": type(e).__name__
        }, session_id)

# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    if not session_id or len(session_id) > 64:
        await websocket.close(code=1008, reason="Invalid session ID")
        return
    
    if not manager.rate_limiter.is_allowed(session_id):
        await websocket.close(code=1008, reason="Rate limit exceeded")
        return
    
    if not await manager.connect(websocket, session_id):
        return
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await manager.send_personal_message(
                    {"type": "ping", "timestamp": datetime.now().isoformat()}, 
                    session_id
                )
                continue
            
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON format"}, 
                    session_id
                )
                continue
            
            manager.session_data[session_id]["last_activity"] = datetime.now()
            await db_manager.update_session_activity(session_id, True)
            message_type = message.get("type")
            
            if message_type == "code_update":
                await handle_code_update(session_id, message)
            elif message_type == "code_execute":
                await handle_code_execution(session_id, message)
            elif message_type == "audio_chunk":
                await handle_audio_chunk(session_id, message)
            elif message_type == "ping":
                await manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.now().isoformat()}, 
                    session_id
                )
            else:
                logger.warning(f"Unknown message type from {session_id}: {message_type}")
                await manager.send_personal_message(
                    {"type": "error", "message": f"Unknown message type: {message_type}"}, 
                    session_id
                )
                
    except WebSocketDisconnect:
        await manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await manager.disconnect(session_id)

# API Endpoints
@app.post("/api/code/execute")
async def execute_code_endpoint(request: CodeExecutionRequest):
    """Execute code using Judge0 API"""
    try:
        if request.test_cases:
            result = await judge0_service.run_test_cases(
                request.code, 
                request.language, 
                [tc.dict() for tc in request.test_cases], 
                request.timeout
            )
        else:
            result = await judge0_service.execute_code(
                request.code, 
                request.language, 
                request.stdin, 
                request.timeout
            )
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Code execution API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/code/analyze")
async def analyze_code_endpoint(request: CodeAnalysisRequest):
    """Analyze code quality and run test cases"""
    try:
        analysis = code_analyzer.analyze_code(request.code, request.language)
        suggestions = generate_enhanced_suggestions(request.code, analysis, request.language)
        
        test_results = None
        if request.test_cases:
            test_results = await judge0_service.run_test_cases(
                request.code,
                request.language,
                [tc.dict() for tc in request.test_cases],
                5
            )
        
        return {
            "success": True,
            "analysis": analysis,
            "suggestions": suggestions,
            "test_results": test_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Code analysis API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/languages")
async def get_supported_languages():
    """Get list of supported programming languages"""
    return {
        "languages": list(SUPPORTED_LANGUAGES.keys()),
        "details": SUPPORTED_LANGUAGES,
        "default": "python"
    }

@app.get("/api/judge0/status")
async def judge0_status():
    """Check Judge0 API status"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{JUDGE0_URL}/about")
            if response.status_code == 200:
                return {
                    "status": "online",
                    "judge0_info": response.json(),
                    "supported_languages": len(SUPPORTED_LANGUAGES),
                    "rate_limit_status": "active"
                }
            else:
                return {
                    "status": "degraded",
                    "message": f"Judge0 returned {response.status_code}"
                }
    except Exception as e:
        return {
            "status": "offline",
            "error": str(e)
        }

@app.get("/api/test/database")
async def test_database():
    try:
        result = await db_manager.test_connection()
        return {"status": "connected", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    
@app.post("/api/session/{session_id}/process-audio")
async def trigger_audio_processing(session_id: str):
    """Manually trigger audio processing for a session"""
    try:
        if session_id not in manager.session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        response = await db_manager.get_audio_chunks_metadata(session_id)
        pending_chunks = [chunk for chunk in response if chunk.get('processing_status') == 'pending']
        
        if not pending_chunks:
            return {"message": "No pending audio chunks found", "processed": 0}
        
        processed_count = 0
        for chunk in pending_chunks[:5]:
            chunk_id = chunk['id']
            success = await process_audio_chunk_from_db(chunk_id)
            if success:
                processed_count += 1
        
        return {
            "message": f"Processed {processed_count}/{len(pending_chunks)} audio chunks",
            "session_id": session_id,
            "processed": processed_count,
            "total_pending": len(pending_chunks)
        }
        
    except Exception as e:
        logger.error(f"Error processing audio for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}/transcripts")
async def get_session_transcripts(session_id: str):
    """Get all transcripts for a session"""
    try:
        transcripts = await db_manager.get_session_transcripts(session_id)
        full_transcript = await db_manager.get_full_session_transcript(session_id)
        
        return {
            "session_id": session_id,
            "transcripts": transcripts,
            "full_transcript": full_transcript,
            "total_chunks": len(transcripts)
        }
        
    except Exception as e:
        logger.error(f"Error getting transcripts for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}/audio-status")
async def get_audio_processing_status(session_id: str):
    """Get audio processing status for a session"""
    try:
        audio_metadata = await db_manager.get_audio_chunks_metadata(session_id)
        
        status_counts = {
            'pending': 0,
            'processing': 0, 
            'completed': 0,
            'failed': 0
        }
        
        for chunk in audio_metadata:
            status = chunk.get('processing_status', 'pending')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "session_id": session_id,
            "total_chunks": len(audio_metadata),
            "status_breakdown": status_counts,
            "chunks": audio_metadata
        }
        
    except Exception as e:
        logger.error(f"Error getting audio status for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/process-all-pending")
async def process_all_pending_audio():
    """Admin endpoint to process all pending audio chunks"""
    try:
        await process_pending_audio_chunks()
        return {"message": "Background audio processing triggered"}
    except Exception as e:
        logger.error(f"Error triggering background processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_chunk/{chunk_id}")
async def process_chunk_transcript(chunk_id: str, session_id: str = None):
    """Generate transcript for a given audio chunk"""
    try:
        logger.info(f"Processing transcript for chunk {chunk_id}")
        
        if not session_id:
            try:
                from database_manager import supabase
                if supabase:
                    response = supabase.table('audio_chunks').select('session_id').eq('id', chunk_id).single().execute()
                    if response.data:
                        session_id = response.data['session_id']
                    else:
                        raise HTTPException(status_code=404, detail=f"Audio chunk {chunk_id} not found")
                else:
                    raise HTTPException(status_code=500, detail="Database not available")
            except Exception as e:
                logger.error(f"Error fetching session_id for chunk {chunk_id}: {e}")
                raise HTTPException(status_code=404, detail=f"Audio chunk {chunk_id} not found")
        
        result = await transcript_service.generate_transcript_for_chunk(chunk_id, session_id)
        
        if result["success"]:
            logger.info(f"Successfully generated transcript for chunk {chunk_id}")
            return {
                "success": True,
                "message": "Transcript generated successfully",
                "data": result
            }
        else:
            logger.error(f"Failed to generate transcript for chunk {chunk_id}: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get('error', 'Failed to generate transcript'))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sandbox_snapshot")
async def get_sandbox_snapshot():
    """Get the most recent sandbox snapshot"""
    try:
        logger.info("Retrieving latest sandbox snapshot")
        
        snapshot = await snapshot_service.get_latest_snapshot()
        
        if snapshot:
            logger.info(f"Retrieved snapshot: {snapshot.get('snapshot_id')}")
            return {
                "success": True,
                "message": "Latest snapshot retrieved successfully",
                "data": snapshot
            }
        else:
            logger.warning("No snapshots available")
            return {
                "success": False,
                "message": "No snapshots available",
                "data": None
            }
            
    except Exception as e:
        logger.error(f"Error retrieving sandbox snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sandbox_snapshot/{snapshot_id}")
async def get_snapshot_by_id(snapshot_id: str):
    """Get a specific snapshot by ID"""
    try:
        logger.info(f"Retrieving snapshot: {snapshot_id}")
        
        snapshot = await snapshot_service.get_snapshot_by_id(snapshot_id)
        
        if snapshot:
            logger.info(f"Retrieved snapshot: {snapshot_id}")
            return {
                "success": True,
                "message": "Snapshot retrieved successfully",
                "data": snapshot
            }
        else:
            logger.warning(f"Snapshot not found: {snapshot_id}")
            raise HTTPException(status_code=404, detail=f"Snapshot {snapshot_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving snapshot {snapshot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sandbox_snapshots")
async def list_snapshots():
    """List all available snapshots"""
    try:
        logger.info("Listing all snapshots")
        
        snapshots = await snapshot_service.list_snapshots()
        
        return {
            "success": True,
            "message": f"Found {len(snapshots)} snapshots",
            "data": {
                "snapshots": snapshots,
                "total_count": len(snapshots)
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing snapshots: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/sandbox_snapshot/trigger")
async def trigger_snapshot():
    """Manually trigger a snapshot creation"""
    try:
        logger.info("Manually triggering snapshot creation")
        
        snapshot = await snapshot_service.take_snapshot()
        
        if snapshot.get("success", True):
            logger.info(f"Snapshot created successfully: {snapshot.get('snapshot_id')}")
            return {
                "success": True,
                "message": "Snapshot created successfully",
                "data": snapshot
            }
        else:
            logger.error(f"Failed to create snapshot: {snapshot.get('error')}")
            raise HTTPException(status_code=500, detail=snapshot.get('error', 'Failed to create snapshot'))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/transcript/{session_id}/{chunk_id}")
async def get_transcript_file(session_id: str, chunk_id: str):
    """Get transcript file information for a specific chunk"""
    try:
        logger.info(f"Getting transcript file for {session_id}/{chunk_id}")
        
        file_info = await transcript_service.get_transcript_file(session_id, chunk_id)
        
        if file_info:
            return {
                "success": True,
                "message": "Transcript file found",
                "data": file_info
            }
        else:
            raise HTTPException(status_code=404, detail="Transcript file not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transcript file for {session_id}/{chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/transcripts/{session_id}")
async def list_session_transcripts(session_id: str):
    """List all transcript files for a session"""
    try:
        logger.info(f"Listing transcripts for session {session_id}")
        
        transcripts = await transcript_service.list_session_transcripts(session_id)
        
        return {
            "success": True,
            "message": f"Found {len(transcripts)} transcript files",
            "data": {
                "session_id": session_id,
                "transcripts": transcripts,
                "total_count": len(transcripts)
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing transcripts for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Enhanced health check with system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(manager.active_connections),
            "total_sessions": len(manager.session_data),
            "system_metrics": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_percent": round(disk.used / disk.total * 100, 2)
            },
            "judge0_integration": "enabled"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/session/{session_id}")
async def get_session_data(session_id: str):
    """Get session data with privacy controls"""
    try:
        if session_id not in manager.session_data:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        
        session_data = manager.session_data[session_id].copy()
        
        if isinstance(session_data.get("connected_at"), datetime):
            session_data["connected_at"] = session_data["connected_at"].isoformat()
            
        if isinstance(session_data.get("last_activity"), datetime):
            session_data["last_activity"] = session_data["last_activity"].isoformat()
            
        if isinstance(session_data.get("disconnected_at"), datetime):
            session_data["disconnected_at"] = session_data["disconnected_at"].isoformat()
            
        if "code_history" in session_data:
            session_data["code_history"] = [
                {k: v for k, v in entry.items() if k != "code"}
                for entry in session_data["code_history"][-10:]
            ]
        
        if "execution_results" in session_data:
            session_data["execution_results"] = [
                {k: v for k, v in entry.items() if k != "code"}
                for entry in session_data["execution_results"][-10:]
            ]
        
        if "audio_chunks" in session_data:
            session_data["audio_chunks"] = [
                {k: v for k, v in entry.items() if k != "audio_data"}
                for entry in session_data["audio_chunks"][-10:]
            ]
        
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session data for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """List sessions with summary information"""
    return {
        "active_sessions": list(manager.active_connections.keys()),
        "total_active": len(manager.active_connections),
        "total_stored": len(manager.session_data),
        "server_capacity": f"{len(manager.active_connections)}/{manager.max_sessions}"
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the home/welcome page"""
    if os.path.exists('templates/home.html'):
        return FileResponse('templates/home.html')
    else:
        return HTMLResponse("""
        <html><head><title>CodeSage AI Interviewer</title></head>
        <body><h1>CodeSage AI Interviewer</h1>
        <p>Welcome! The server is running with Judge0 integration.</p>
        <p><a href="/interview">Start Interview</a> | <a href="/monitor">Monitor</a></p>
        </body></html>
        """)

@app.get("/home", response_class=HTMLResponse)  
async def serve_home():
    """Serve the home page (alias for root)"""
    return await read_root()

@app.get("/interview", response_class=HTMLResponse)
async def serve_interview():
    """Serve the interview interface"""
    if os.path.exists('templates/index.html'):
        return FileResponse('templates/index.html')
    else:
        return HTMLResponse("""
        <html><head><title>Interview Interface</title></head>
        <body><h1>Interview Interface</h1>
        <p>Interview interface would be here. Template file missing.</p>
        <p><a href="/">Back to Home</a></p>
        </body></html>
        """)

@app.get("/monitor")
async def serve_monitor():
    """Serve the monitoring dashboard"""
    if os.path.exists('templates/monitor.html'):
        return FileResponse('templates/monitor.html')
    else:
        return HTMLResponse("""
        <html><head><title>System Monitor</title></head>
        <body><h1>System Monitor</h1>
        <p>Monitor dashboard would be here. Template file missing.</p>
        <p><a href="/health">Health Check</a> | <a href="/sessions">Sessions</a></p>
        </body></html>
        """)

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests"""
    return Response(status_code=204)

@app.get("/api/session/{session_id}/audio/{chunk_index}")
async def get_audio_chunk(session_id: str, chunk_index: int):
    """Retrieve a specific audio chunk from the database"""
    try:
        if session_id not in manager.session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if chunk_index < 0:
            raise HTTPException(status_code=400, detail="Invalid chunk index")
        
        try:
            audio_data = await db_manager.get_audio_chunk(session_id, chunk_index)
        except Exception as e:
            logger.error(f"Database error retrieving audio chunk {chunk_index} for session {session_id}: {e}")
            raise HTTPException(status_code=500, detail="Database error")
        
        if not audio_data:
            raise HTTPException(status_code=404, detail="Audio chunk not found")
        
        audio_format = audio_data.get('format', 'webm')
        audio_base64 = audio_data.get('audio_data', '')
        timestamp = audio_data.get('timestamp', '')
        duration = audio_data.get('duration', 0)
        
        if not audio_base64:
            raise HTTPException(status_code=404, detail="Audio data not found")
        
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            logger.error(f"Failed to decode audio data for chunk {chunk_index}: {e}")
            raise HTTPException(status_code=500, detail="Invalid audio data encoding")
        
        content_type_map = {
            'webm': 'audio/webm',
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'ogg': 'audio/ogg'
        }
        content_type = content_type_map.get(audio_format.lower(), 'audio/webm')
        
        response = Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                'Content-Disposition': f'inline; filename="audio_{session_id}_{chunk_index}.{audio_format}"',
                'Cache-Control': 'private, max-age=3600',
                'X-Audio-Duration': str(duration),
                'X-Audio-Timestamp': timestamp,
                'X-Audio-Format': audio_format,
                'X-Chunk-Index': str(chunk_index),
                'X-Session-ID': session_id
            }
        )
        
        logger.info(f"Served audio chunk {chunk_index} for session {session_id} ({len(audio_bytes)} bytes, {audio_format})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving audio chunk {chunk_index} for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/sessions/detailed")
async def get_detailed_sessions():
    """Get detailed information about all sessions"""
    detailed_sessions = {}
    
    for session_id, session_data in manager.session_data.items():
        is_active = session_id in manager.active_connections
        
        code_history = session_data.get("code_history", [])
        execution_results = session_data.get("execution_results", [])
        audio_chunks = session_data.get("audio_chunks", [])
        
        latest_quality_score = 0
        if code_history:
            latest_code = code_history[-1].get("code", "")
            if latest_code:
                analysis = code_analyzer.analyze_code(latest_code)
                latest_quality_score = analysis.get("code_quality_score", 0)
        
        detailed_sessions[session_id] = {
            "session_id": session_id,
            "is_active": is_active,
            "connection_info": {
                "connected_at": session_data.get("connected_at", "").isoformat() if isinstance(session_data.get("connected_at"), datetime) else session_data.get("connected_at", ""),
                "last_activity": session_data.get("last_activity", "").isoformat() if isinstance(session_data.get("last_activity"), datetime) else session_data.get("last_activity", ""),
                "disconnected_at": session_data.get("disconnected_at", "").isoformat() if session_data.get("disconnected_at") and isinstance(session_data.get("disconnected_at"), datetime) else session_data.get("disconnected_at")
            },
            "statistics": {
                "total_code_updates": len(code_history),
                "total_executions": session_data.get("execution_count", 0),
                "total_audio_chunks": len(audio_chunks),
                "total_audio_duration": session_data.get("total_audio_duration", 0),
                "current_code_length": len(code_history[-1]["code"]) if code_history else 0,
                "latest_quality_score": latest_quality_score
            },
            "resource_usage": session_data.get("resource_usage", {}),
            "security_level": "Judge0 Sandboxed"
        }
    
    return {
        "sessions": detailed_sessions,
        "total_active": len(manager.active_connections),
        "total_stored": len(manager.session_data),
        "server_stats": {
            "judge0_integration": "enabled",
            "rate_limiter_active": True,
            "cleanup_enabled": True
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    logger.info("Starting CodeSage AI Interviewer Enhanced Server...")
    logger.info(f"Judge0 integration: Enabled")
    logger.info(f"Max concurrent sessions: {manager.max_sessions}")
    logger.info(f"Session TTL: {manager.session_ttl}")
    logger.info(f"Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True,
        workers=1
    )