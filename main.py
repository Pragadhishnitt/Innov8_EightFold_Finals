import asyncio
import json
import logging
import base64
import subprocess
import sys
import tempfile
import os
import time
import hashlib
from typing import Dict, List, Optional, Any
from fastapi import Response
import asyncio
from audio_processor import process_audio_chunk_from_db, process_pending_audio_chunks
from transcript_service import transcript_service
from snapshot_service import snapshot_service
# Try to import docker, but don't fail if it's not available
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None
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

from database_manager import db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security and validation models
security = HTTPBearer(auto_error=False)

class CodeExecutionRequest(BaseModel):
    code: str = Field(..., max_length=50000)
    language: str = Field(default="python", pattern="^(python|javascript|java)$")
    timeout: int = Field(default=5, ge=1, le=30)
    
    @field_validator('code')
    @classmethod
    def validate_code_safety(cls, v):
        # Basic security checks for dangerous operations
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess', '__import__',
            'eval(', 'exec(', 'compile(', 'open(', 'file(',
            'input(', 'raw_input(', 'urllib', 'requests', 'socket'
        ]
        
        code_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                raise ValueError(f'Potentially unsafe operation detected: {pattern}')
        return v

class AudioChunkRequest(BaseModel):
    audio_data: str = Field(..., max_length=1000000)  # Limit audio size
    format: str = Field(default="webm", pattern="^(webm|mp3|wav|ogg)$")
    duration: float = Field(default=0, ge=0, le=300)  # Max 5 minutes

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
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.window
        ]
        
        # Check if under limit
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
        # Store session in database
        await db_manager.store_session(session_id)
        return True

    async def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        # Keep session data for a while for analysis
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
        
        # Clean up disconnected clients
        for session_id in disconnected:
            await self.disconnect(session_id)
    
    async def cleanup_old_sessions(self):
        """Remove old inactive sessions"""
        cutoff_time = datetime.now() - self.session_ttl
        to_remove = []
        
        for session_id, data in self.session_data.items():
            if (session_id not in self.active_connections and 
                data.get("last_activity", datetime.min) < cutoff_time):
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.session_data[session_id]
            logger.info(f"Cleaned up old session: {session_id}")

# Enhanced Secure Code Sandbox
class SecureCodeSandbox:
    def __init__(self):
        self.docker_client = None
        self.fallback_mode = True
        
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                # Test Docker connectivity
                self.docker_client.ping()
                self.fallback_mode = False
                logger.info("Docker sandbox initialized successfully")
            except Exception as e:
                logger.warning(f"Docker not available, using restricted fallback mode: {e}")
        else:
            logger.info("Docker module not installed, using restricted fallback mode")
    
    def execute_python_code(self, code: str, timeout: int = 5) -> dict:
        """Execute Python code in a secure sandbox"""
        if not self.fallback_mode and self.docker_client:
            return self._execute_in_docker(code, timeout)
        else:
            return self._execute_restricted_fallback(code, timeout)
    
    def _execute_in_docker(self, code: str, timeout: int) -> dict:
        """Execute code in Docker container"""
        try:
            # Create a secure Docker container
            container = self.docker_client.containers.run(
                "python:3.9-alpine",
                f"python -c '{code}'",
                detach=True,
                mem_limit="128m",
                cpu_period=100000,
                cpu_quota=50000,  # 50% CPU limit
                network_disabled=True,
                remove=True,
                stdout=True,
                stderr=True
            )
            
            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            logs = container.logs().decode('utf-8')
            
            return {
                "success": result['StatusCode'] == 0,
                "output": logs if result['StatusCode'] == 0 else "",
                "error": logs if result['StatusCode'] != 0 else "",
                "execution_time": f"< {timeout}s",
                "memory_usage": "< 128MB",
                "security_level": "Docker Sandboxed"
            }
            
        except docker.errors.ContainerError as e:
            return {
                "success": False,
                "output": "",
                "error": f"Container error: {str(e)}",
                "execution_time": f"< {timeout}s",
                "memory_usage": "N/A",
                "security_level": "Docker Sandboxed"
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Docker execution failed: {str(e)}",
                "execution_time": "N/A",
                "memory_usage": "N/A",
                "security_level": "Docker Sandboxed"
            }
    
    def _execute_restricted_fallback(self, code: str, timeout: int) -> dict:
        """Fallback execution with heavy restrictions"""
        try:
            # Additional safety checks
            if any(keyword in code.lower() for keyword in 
                   ['import', 'exec', 'eval', 'open', 'file', '__']):
                return {
                    "success": False,
                    "output": "",
                    "error": "Restricted operations not allowed in fallback mode",
                    "execution_time": "0s",
                    "memory_usage": "0MB",
                    "security_level": "Restricted Fallback"
                }
            
            # Create temporary file in secure location
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Wrap code in restricted environment
                restricted_code = f"""
import sys
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timeout")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

try:
    # User code here
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    print(f"Error: {{e}}")
finally:
    signal.alarm(0)
"""
                f.write(restricted_code)
                temp_file = f.name
            
            # Execute with strict limitations
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir(),
                env={"PYTHONDONTWRITEBYTECODE": "1"}  # Prevent .pyc files
            )
            
            # Clean up
            os.unlink(temp_file)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout[:1000],  # Limit output
                "error": result.stderr[:1000] if result.stderr else "",
                "execution_time": f"< {timeout}s",
                "memory_usage": "Limited",
                "security_level": "Restricted Fallback"
            }
            
        except subprocess.TimeoutExpired:
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return {
                "success": False,
                "output": "",
                "error": "Code execution timed out",
                "execution_time": f"> {timeout}s",
                "memory_usage": "N/A",
                "security_level": "Restricted Fallback"
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "execution_time": "N/A",
                "memory_usage": "N/A",
                "security_level": "Restricted Fallback"
            }

# Enhanced Code Analysis
class CodeAnalyzer:
    @staticmethod
    def analyze_code(code: str) -> dict:
        """Comprehensive code analysis"""
        analysis = {
            "syntax_valid": True,
            "complexity": "Low",
            "style_issues": [],
            "potential_errors": [],
            "security_issues": [],
            "performance_hints": [],
            "code_quality_score": 100
        }
        
        try:
            # Syntax validation
            compile(code, '<string>', 'exec')
            analysis["syntax_valid"] = True
        except SyntaxError as e:
            analysis["syntax_valid"] = False
            analysis["potential_errors"].append(f"Syntax Error: {str(e)}")
            analysis["code_quality_score"] -= 30
        
        # Complexity analysis
        nested_loops = 0
        lines = code.split('\n')
        indent_level = 0
        max_indent = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
                
            # Count indentation
            current_indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, current_indent)
            
            # Count nested structures
            if any(keyword in stripped for keyword in ['for ', 'while ']):
                nested_loops += 1
        
        if nested_loops >= 3:
            analysis["complexity"] = "High"
            analysis["performance_hints"].append("Consider reducing nested loops")
            analysis["code_quality_score"] -= 20
        elif nested_loops >= 2:
            analysis["complexity"] = "Medium"
            analysis["performance_hints"].append("Watch out for O(n²) complexity")
            analysis["code_quality_score"] -= 10
        
        # Style analysis
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                analysis["style_issues"].append(f"Line {i}: Line too long (>120 chars)")
                analysis["code_quality_score"] -= 2
            
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                continue  # Top level is fine
            elif line.strip() and '    ' not in line and '\t' not in line and i > 1:
                analysis["style_issues"].append(f"Line {i}: Inconsistent indentation")
                analysis["code_quality_score"] -= 5
        
        # Security analysis
        security_patterns = {
            'eval(': 'Dangerous eval() usage',
            'exec(': 'Dangerous exec() usage', 
            '__import__': 'Dynamic imports detected',
            'input(': 'User input without validation'
        }
        
        code_lower = code.lower()
        for pattern, message in security_patterns.items():
            if pattern in code_lower:
                analysis["security_issues"].append(message)
                analysis["code_quality_score"] -= 15
        
        # Ensure score doesn't go below 0
        analysis["code_quality_score"] = max(0, analysis["code_quality_score"])
        
        return analysis
    
    @staticmethod
    def generate_suggestions(code: str, analysis: dict) -> List[str]:
        """Generate intelligent suggestions based on analysis"""
        suggestions = []
        
        if not analysis["syntax_valid"]:
            suggestions.append("🔧 Fix syntax errors before proceeding")
        
        if analysis["complexity"] == "High":
            suggestions.append("⚡ Consider optimizing algorithm complexity - maybe use hash maps or sort first?")
        elif analysis["complexity"] == "Medium":
            suggestions.append("💡 Good structure! Consider if you can optimize any nested loops")
        
        if analysis["security_issues"]:
            suggestions.append("🛡️ Security concerns detected - avoid eval/exec and validate inputs")
        
        if not code.strip():
            suggestions.append("🚀 Start by defining your function signature and thinking about the approach")
        elif len(code.strip().split('\n')) < 3:
            suggestions.append("📝 Consider adding comments to explain your approach")
        
        if analysis["code_quality_score"] > 80:
            suggestions.append("✨ Great code quality! Keep it up!")
        elif analysis["code_quality_score"] > 60:
            suggestions.append("👍 Good progress! Address the style issues for cleaner code")
        else:
            suggestions.append("🔨 Focus on fixing the issues highlighted above")
        
        return suggestions

# Initialize components
manager = ConnectionManager()
sandbox = SecureCodeSandbox()
analyzer = CodeAnalyzer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    async def cleanup_task():
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            await manager.cleanup_old_sessions()
    
    async def audio_processing_task():
        while True:
            await asyncio.sleep(60)  # Every minute
            await process_pending_audio_chunks()
    
    async def snapshot_task():
        """Background task for periodic sandbox snapshots"""
        await snapshot_service.start_background_snapshot_task()
    
    cleanup_task_handle = asyncio.create_task(cleanup_task())
    audio_task_handle = asyncio.create_task(audio_processing_task())
    snapshot_task_handle = asyncio.create_task(snapshot_task())
    
    yield
    
    # Shutdown
    cleanup_task_handle.cancel()
    audio_task_handle.cancel()
    snapshot_task_handle.cancel()
    snapshot_service.stop_background_snapshot_task()
# Initialize FastAPI app
app = FastAPI(
    title="CodeSage AI Interviewer - Enhanced", 
    version="2.0.0",
    description="Secure AI-powered coding interview platform with real-time monitoring",
    lifespan=lifespan
)

# FIXED: Enhanced CORS configuration with localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:8000",  # Added this - same port as server
        "http://localhost:8080", 
        "http://localhost:8081",  
        "http://127.0.0.1:8000",  # Added this - IP version
        "http://127.0.0.1:8081",
        "file://"  # For direct file access
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# Authentication dependency (basic implementation)
async def get_current_session(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    # In production, validate JWT token here
    return credentials.credentials


# WebSocket endpoint with enhanced security
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    # Validate session ID format
    if not session_id or len(session_id) > 64:
        await websocket.close(code=1008, reason="Invalid session ID")
        return
    
    # Rate limiting check
    if not manager.rate_limiter.is_allowed(session_id):
        await websocket.close(code=1008, reason="Rate limit exceeded")
        return
    
    if not await manager.connect(websocket, session_id):
        return
    
    try:
        while True:
            # Receive message with timeout
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
            
            # Update activity timestamp
            manager.session_data[session_id]["last_activity"] = datetime.now()
            # Update database activity
            await db_manager.update_session_activity(session_id, True)
            message_type = message.get("type")
            
            # Route messages to handlers
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

# Enhanced message handlers
async def handle_code_update(session_id: str, message: dict):
    """Handle real-time code updates with validation"""
    try:
        # Validate using Pydantic model
        update_data = CodeUpdateRequest(**message)
        
        code = update_data.code
        cursor_position = update_data.cursor_position
        timestamp = datetime.now()
        
        # Store code history with size limits
        code_entry = {
            "code": code,
            "cursor_position": cursor_position,
            "timestamp": timestamp.isoformat(),
            "line_count": len(code.split('\n')),
            "char_count": len(code),
            "hash": hashlib.md5(code.encode()).hexdigest()
        }
        
        session_data = manager.session_data[session_id]
        session_data["code_history"].append(code_entry)
        await db_manager.store_code_update(session_id, code_entry)

        # Keep only last 50 code updates to prevent memory bloat
        if len(session_data["code_history"]) > 50:
            session_data["code_history"] = session_data["code_history"][-50:]
        
        # Analyze code
        analysis = analyzer.analyze_code(code)
        suggestions = analyzer.generate_suggestions(code, analysis)
        
        # Send analysis back to client
        response = {
            "type": "code_analysis",
            "timestamp": timestamp.isoformat(),
            "analysis": analysis,
            "suggestions": suggestions,
            "session_stats": {
                "total_updates": len(session_data["code_history"]),
                "current_quality_score": analysis["code_quality_score"]
            }
        }
        
        await manager.send_personal_message(response, session_id)
        
    except Exception as e:
        logger.error(f"Code update error for session {session_id}: {e}")
        await manager.send_personal_message(
            {"type": "error", "message": "Failed to process code update"}, 
            session_id
        )

async def handle_code_execution(session_id: str, message: dict):
    """Handle code execution with enhanced security"""
    try:
        # Rate limiting for executions
        session_data = manager.session_data[session_id]
        if session_data["execution_count"] >= 10:  # Max 10 executions per session
            await manager.send_personal_message({
                "type": "error", 
                "message": "Execution limit reached for this session"
            }, session_id)
            return
        
        # Validate using Pydantic model
        exec_data = CodeExecutionRequest(**message)
        
        code = exec_data.code
        language = exec_data.language
        timeout = exec_data.timeout
        
        # Execute code
        if language == "python":
            execution_result = sandbox.execute_python_code(code, timeout)
        else:
            execution_result = {
                "success": False,
                "output": "",
                "error": f"Language '{language}' not supported yet",
                "execution_time": "N/A",
                "memory_usage": "N/A",
                "security_level": "N/A"
            }
        
        # Store execution result
        result_entry = {
            "code": code[:500],  # Store limited code for privacy
            "code_hash": hashlib.md5(code.encode()).hexdigest(),
            "result": execution_result,
            "timestamp": datetime.now().isoformat(),
            "language": language
        }
        
        session_data["execution_results"].append(result_entry)
        await db_manager.store_execution_result(session_id, result_entry)

        session_data["execution_count"] += 1
        
        # Keep only last 20 execution results
        if len(session_data["execution_results"]) > 20:
            session_data["execution_results"] = session_data["execution_results"][-20:]
        
        # Send result back to client
        response = {
            "type": "execution_result",
            "timestamp": result_entry["timestamp"],
            "result": execution_result,
            "session_stats": {
                "total_executions": session_data["execution_count"],
                "executions_remaining": 10 - session_data["execution_count"]
            }
        }
        
        await manager.send_personal_message(response, session_id)
        
    except ValueError as e:
        await manager.send_personal_message({
            "type": "error", 
            "message": f"Invalid code: {str(e)}"
        }, session_id)
    except Exception as e:
        logger.error(f"Code execution error for session {session_id}: {e}")
        await manager.send_personal_message({
            "type": "error", 
            "message": "Failed to execute code"
        }, session_id)

async def handle_audio_chunk(session_id: str, message: dict):
    """ENHANCED AND FIXED: Audio chunk handler with comprehensive error handling"""
    try:
        logger.info(f"Processing audio chunk for session {session_id}")
        logger.info(f"Message keys: {list(message.keys())}")
        
        # Extract and validate audio data
        if 'audio_data' not in message:
            logger.error("No 'audio_data' key in message!")
            await manager.send_personal_message({
                "type": "error",
                "message": "No audio_data field in message",
                "debug_info": {"received_keys": list(message.keys())}
            }, session_id)
            return
        
        audio_data_raw = message.get('audio_data')
        audio_format = message.get("format", "webm")
        duration = float(message.get("duration", 0))
        
        # Handle data URL format (data:audio/webm;base64,XXXXX)
        if isinstance(audio_data_raw, str) and audio_data_raw.startswith('data:'):
            try:
                # Extract base64 part after comma
                _, audio_data_b64 = audio_data_raw.split(',', 1)
                logger.info(f"Extracted base64 from data URL, length: {len(audio_data_b64)}")
            except ValueError:
                logger.error("Invalid data URL format")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid data URL format"
                }, session_id)
                return
        else:
            audio_data_b64 = str(audio_data_raw)
        
        # Validate base64 data
        if not audio_data_b64 or len(audio_data_b64) < 100:
            logger.error(f"Audio data too short: {len(audio_data_b64)} chars")
            await manager.send_personal_message({
                "type": "error",
                "message": f"Audio data too short: {len(audio_data_b64)} characters"
            }, session_id)
            return
        
        # Test base64 decoding
        try:
            decoded = base64.b64decode(audio_data_b64)
            logger.info(f"Successfully decoded {len(decoded)} bytes of audio")
            
            if len(decoded) < 1000:  # Very small audio file
                logger.warning(f"Decoded audio very small: {len(decoded)} bytes")
        except Exception as e:
            logger.error(f"Base64 decode failed: {e}")
            await manager.send_personal_message({
                "type": "error",
                "message": f"Invalid base64 audio data: {str(e)}"
            }, session_id)
            return
        
        # Validate session exists
        session_data = manager.session_data.get(session_id)
        if not session_data:
            logger.error(f"Session data not found for {session_id}")
            await manager.send_personal_message({
                "type": "error",
                "message": "Session not found"
            }, session_id)
            return
        
        # Check duration limits
        total_duration = session_data.get("total_audio_duration", 0)
        if total_duration + duration > 1800:  # 30 minutes max
            await manager.send_personal_message({
                "type": "error",
                "message": f"Audio duration limit exceeded (current: {total_duration}s)"
            }, session_id)
            return
        
        # Prepare audio entry with proper structure
        timestamp = datetime.now()
        audio_entry = {
            "format": audio_format,
            "timestamp": timestamp.isoformat(),
            "duration": duration,
            "size_bytes": len(audio_data_b64),
            "audio_hash": hashlib.md5(audio_data_b64.encode()).hexdigest(),
            "audio_data": audio_data_b64,
            "chunk_index": len(session_data.get("audio_chunks", []))  # Add sequential index
        }
        
        logger.info(f"Storing audio chunk - Format: {audio_format}, Duration: {duration}s, Size: {len(audio_data_b64)} chars")
        
        # Store in database with enhanced error handling
        try:
            chunk_id = await db_manager.store_audio_chunk(session_id, audio_entry)
            logger.info(f"Database store operation result: {chunk_id}")
            
            if chunk_id is None:
                raise Exception("Database returned None for chunk_id")
            
            if not isinstance(chunk_id, (int, str)) or (isinstance(chunk_id, str) and not chunk_id.strip()):
                raise Exception(f"Invalid chunk_id returned: {chunk_id} (type: {type(chunk_id)})")
                
        except Exception as db_error:
            logger.error(f"Database storage failed: {db_error}", exc_info=True)
            await manager.send_personal_message({
                "type": "error",
                "message": f"Database storage failed: {str(db_error)}"
            }, session_id)
            return
        
        # Successful storage
        logger.info(f"Successfully stored audio chunk with ID: {chunk_id}")
        
        # Update session tracking
        session_data["audio_chunks"].append({
            "chunk_id": chunk_id,
            "timestamp": timestamp.isoformat(),
            "duration": duration,
            "format": audio_format
        })
        session_data["total_audio_duration"] = total_duration + duration
        
        # Trigger processing
        logger.info(f"Triggering audio processing for chunk {chunk_id}")
        asyncio.create_task(process_audio_chunk_from_db(str(chunk_id)))
        
        # Send success response
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
        
        logger.info(f"Audio chunk processing completed successfully for session {session_id}")
        
    except Exception as e:
        logger.error(f"Critical error in handle_audio_chunk: {e}", exc_info=True)
        await manager.send_personal_message({
            "type": "error",
            "message": f"Failed to process audio: {str(e)}",
            "error_type": type(e).__name__
        }, session_id)

# API Endpoints

# Add this test endpoint to main.py
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
        
        # Get pending chunks for this session
        response = await db_manager.get_audio_chunks_metadata(session_id)
        pending_chunks = [chunk for chunk in response if chunk.get('processing_status') == 'pending']
        
        if not pending_chunks:
            return {"message": "No pending audio chunks found", "processed": 0}
        
        # Process chunks
        processed_count = 0
        for chunk in pending_chunks[:5]:  # Process max 5 at a time
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

# New endpoints for transcript generation and sandbox snapshots

@app.post("/process_chunk/{chunk_id}")
async def process_chunk_transcript(chunk_id: str, session_id: str = None):
    """
    Generate transcript for a given audio chunk.
    
    Args:
        chunk_id: The ID of the audio chunk to process
        session_id: Optional session ID (will be fetched from database if not provided)
    
    Returns:
        Dict containing transcript information and file path
    """
    try:
        logger.info(f"Processing transcript for chunk {chunk_id}")
        
        # If session_id not provided, fetch it from database
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
        
        # Generate transcript
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
    """
    Get the most recent sandbox snapshot.
    
    Returns:
        Dict containing the latest snapshot data
    """
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
    """
    Get a specific snapshot by ID.
    
    Args:
        snapshot_id: The ID of the snapshot to retrieve
    
    Returns:
        Dict containing the snapshot data
    """
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
    """
    List all available snapshots.
    
    Returns:
        Dict containing list of all snapshots
    """
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
    """
    Manually trigger a snapshot creation.
    
    Returns:
        Dict containing the new snapshot data
    """
    try:
        logger.info("Manually triggering snapshot creation")
        
        snapshot = await snapshot_service.take_snapshot()
        
        if snapshot.get("success", True):  # Default to True if success key not present
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
    """
    Get transcript file information for a specific chunk.
    
    Args:
        session_id: The session ID
        chunk_id: The chunk ID
    
    Returns:
        Dict containing transcript file information
    """
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
    """
    List all transcript files for a session.
    
    Args:
        session_id: The session ID
    
    Returns:
        Dict containing list of transcript files
    """
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
            "sandbox_mode": "docker" if not sandbox.fallback_mode else "restricted_fallback"
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
        
        # Handle datetime conversion safely
        if isinstance(session_data.get("connected_at"), datetime):
            session_data["connected_at"] = session_data["connected_at"].isoformat()
            
        if isinstance(session_data.get("last_activity"), datetime):
            session_data["last_activity"] = session_data["last_activity"].isoformat()
            
        if isinstance(session_data.get("disconnected_at"), datetime):
            session_data["disconnected_at"] = session_data["disconnected_at"].isoformat()
            
        # Remove sensitive data
        if "code_history" in session_data:
            session_data["code_history"] = [
                {k: v for k, v in entry.items() if k != "code"}
                for entry in session_data["code_history"][-10:]  # Last 10 only
            ]
        
        if "execution_results" in session_data:
            session_data["execution_results"] = [
                {k: v for k, v in entry.items() if k != "code"}
                for entry in session_data["execution_results"][-10:]
            ]
        
        # Remove actual audio data
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

# FIXED: Single set of routes without duplicates
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the home/welcome page"""
    return FileResponse('templates/home.html')

@app.get("/home", response_class=HTMLResponse)  
async def serve_home():
    """Serve the home page (alias for root)"""
    return FileResponse('templates/home.html')

@app.get("/interview", response_class=HTMLResponse)
async def serve_interview():
    """Serve the interview interface"""
    return FileResponse('templates/index.html')

@app.get("/monitor")
async def serve_monitor():
    """Serve the monitoring dashboard"""
    return FileResponse('templates/monitor.html')

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests"""
    return Response(status_code=204)

# Audio retrieval endpoints
@app.get("/api/session/{session_id}/audio/{chunk_index}")
async def get_audio_chunk(session_id: str, chunk_index: int):
    """Retrieve a specific audio chunk from the database"""
    try:
        # Validate session exists
        if session_id not in manager.session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate chunk index
        if chunk_index < 0:
            raise HTTPException(status_code=400, detail="Invalid chunk index")
        
        # Get audio chunk from database
        try:
            audio_data = await db_manager.get_audio_chunk(session_id, chunk_index)
        except Exception as e:
            logger.error(f"Database error retrieving audio chunk {chunk_index} for session {session_id}: {e}")
            raise HTTPException(status_code=500, detail="Database error")
        
        if not audio_data:
            raise HTTPException(status_code=404, detail="Audio chunk not found")
        
        # Extract audio information
        audio_format = audio_data.get('format', 'webm')
        audio_base64 = audio_data.get('audio_data', '')
        timestamp = audio_data.get('timestamp', '')
        duration = audio_data.get('duration', 0)
        
        if not audio_base64:
            raise HTTPException(status_code=404, detail="Audio data not found")
        
        # Decode base64 audio data
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            logger.error(f"Failed to decode audio data for chunk {chunk_index}: {e}")
            raise HTTPException(status_code=500, detail="Invalid audio data encoding")
        
        # Determine content type based on format
        content_type_map = {
            'webm': 'audio/webm',
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'ogg': 'audio/ogg'
        }
        content_type = content_type_map.get(audio_format.lower(), 'audio/webm')
        
        # Create response with proper headers
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

# Additional API endpoints for monitoring and analytics
@app.get("/api/sessions/detailed")
async def get_detailed_sessions():
    """Get detailed information about all sessions"""
    detailed_sessions = {}
    
    for session_id, session_data in manager.session_data.items():
        is_active = session_id in manager.active_connections
        
        # Calculate session metrics
        code_history = session_data.get("code_history", [])
        execution_results = session_data.get("execution_results", [])
        audio_chunks = session_data.get("audio_chunks", [])
        
        latest_quality_score = 0
        if code_history:
            latest_code = code_history[-1].get("code", "")
            if latest_code:
                analysis = analyzer.analyze_code(latest_code)
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
            "security_level": "Docker Sandboxed" if not sandbox.fallback_mode else "Restricted Fallback"
        }
    
    return {
        "sessions": detailed_sessions,
        "total_active": len(manager.active_connections),
        "total_stored": len(manager.session_data),
        "server_stats": {
            "sandbox_mode": "docker" if not sandbox.fallback_mode else "restricted_fallback",
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
    logger.info(f"Sandbox mode: {'Docker' if not sandbox.fallback_mode else 'Restricted Fallback'}")
    logger.info(f"Max concurrent sessions: {manager.max_sessions}")
    logger.info(f"Session TTL: {manager.session_ttl}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload in production
        log_level="info",
        access_log=True,
        workers=1  # WebSocket applications should use single worker
    )