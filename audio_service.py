import asyncio
import json
import logging
import base64
import os
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
from collections import defaultdict

# Import your existing audio processor
from audio_processor import process_audio_chunk_from_db, transcribe_with_gemini, process_pending_audio_chunks
from database_manager import db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioData(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio data")
    format: str = Field(default="webm", description="Audio format")
    duration: float = Field(default=0.0, description="Duration in seconds")
    session_id: str = Field(default="default", description="Session identifier")

class AudioProcessingService:
    def __init__(self):
        self.active_sessions: Dict[str, WebSocket] = {}
        self.processing_queue: List[Dict] = []
        self.transcripts: Dict[str, List[Dict]] = defaultdict(list)
    
    async def add_to_queue(self, session_id: str, audio_data: str, format: str, duration: float):
        """Add audio data to processing queue"""
        # Store in database first
        audio_entry = {
            "audio_data": audio_data,
            "format": format,
            "duration": duration,
            "size_bytes": len(audio_data),
            "audio_hash": f"hash_{datetime.now().timestamp()}"
        }
        
        chunk_id = await db_manager.store_audio_chunk(session_id, audio_entry)
        
        if chunk_id:
            # Add to processing queue
            queue_item = {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "status": "queued"
            }
            self.processing_queue.append(queue_item)
            logger.info(f"Added audio chunk {chunk_id} to processing queue for session {session_id}")
            
            # Trigger processing
            asyncio.create_task(self.process_audio_chunk(chunk_id, session_id))
            return chunk_id
        
        return None
    
    async def process_audio_chunk(self, chunk_id: int, session_id: str):
        """Process a single audio chunk"""
        try:
            logger.info(f"Starting processing for chunk {chunk_id}")
            
            # Update queue status
            for item in self.processing_queue:
                if item["chunk_id"] == chunk_id:
                    item["status"] = "processing"
                    break
            
            # Send status update to connected clients
            await self.broadcast_to_session(session_id, {
                "type": "processing_status",
                "chunk_id": chunk_id,
                "status": "processing",
                "timestamp": datetime.now().isoformat()
            })
            
            # Process the audio
            success = await process_audio_chunk_from_db(chunk_id)
            
            if success:
                # Get the transcript
                transcripts = await db_manager.get_session_transcripts(session_id)
                latest_transcript = transcripts[-1] if transcripts else None
                
                if latest_transcript:
                    # Store in memory for quick access
                    self.transcripts[session_id].append({
                        "chunk_id": chunk_id,
                        "transcript": latest_transcript.get("transcript", ""),
                        "confidence": latest_transcript.get("transcript_confidence", 0.0),
                        "timestamp": latest_transcript.get("created_at", ""),
                        "duration": latest_transcript.get("duration", 0.0)
                    })
                    
                    # Send transcript to connected clients
                    await self.broadcast_to_session(session_id, {
                        "type": "transcript_ready",
                        "chunk_id": chunk_id,
                        "transcript": latest_transcript.get("transcript", ""),
                        "confidence": latest_transcript.get("transcript_confidence", 0.0),
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Update queue status
                for item in self.processing_queue:
                    if item["chunk_id"] == chunk_id:
                        item["status"] = "completed"
                        break
                        
                logger.info(f"Successfully processed chunk {chunk_id}")
            else:
                # Mark as failed
                for item in self.processing_queue:
                    if item["chunk_id"] == chunk_id:
                        item["status"] = "failed"
                        break
                
                await self.broadcast_to_session(session_id, {
                    "type": "processing_error",
                    "chunk_id": chunk_id,
                    "error": "Failed to process audio chunk",
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.error(f"Failed to process chunk {chunk_id}")
        
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            await self.broadcast_to_session(session_id, {
                "type": "processing_error",
                "chunk_id": chunk_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
    
    async def connect_session(self, session_id: str, websocket: WebSocket):
        """Connect a WebSocket for a session"""
        await websocket.accept()
        self.active_sessions[session_id] = websocket
        logger.info(f"Audio service: Session {session_id} connected")
        
        # Send existing transcripts
        if session_id in self.transcripts:
            await websocket.send_text(json.dumps({
                "type": "existing_transcripts",
                "transcripts": self.transcripts[session_id],
                "timestamp": datetime.now().isoformat()
            }))
    
    async def disconnect_session(self, session_id: str):
        """Disconnect a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        logger.info(f"Audio service: Session {session_id} disconnected")
    
    async def broadcast_to_session(self, session_id: str, message: dict):
        """Send message to a specific session"""
        if session_id in self.active_sessions:
            try:
                await self.active_sessions[session_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to session {session_id}: {e}")
                await self.disconnect_session(session_id)

# Initialize service
audio_service = AudioProcessingService()

# Create FastAPI app
app = FastAPI(
    title="Audio Processing Service",
    version="1.0.0",
    description="Standalone audio transcription service"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8001",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

# WebSocket endpoint for real-time communication
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await audio_service.connect_session(session_id, websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "audio_chunk":
                # Process incoming audio
                chunk_id = await audio_service.add_to_queue(
                    session_id,
                    message.get("audio_data", ""),
                    message.get("format", "webm"),
                    message.get("duration", 0.0)
                )
                
                await websocket.send_text(json.dumps({
                    "type": "audio_received",
                    "chunk_id": chunk_id,
                    "status": "queued" if chunk_id else "failed",
                    "timestamp": datetime.now().isoformat()
                }))
                
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
    
    except WebSocketDisconnect:
        await audio_service.disconnect_session(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        await audio_service.disconnect_session(session_id)

# REST endpoints
@app.post("/api/process-audio")
async def process_audio(audio_data: AudioData):
    """Process audio via REST API"""
    try:
        chunk_id = await audio_service.add_to_queue(
            audio_data.session_id,
            audio_data.audio_data,
            audio_data.format,
            audio_data.duration
        )
        
        if chunk_id:
            return {
                "success": True,
                "chunk_id": chunk_id,
                "message": "Audio queued for processing",
                "session_id": audio_data.session_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to queue audio for processing")
    
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/transcripts/{session_id}")
async def get_transcripts(session_id: str):
    """Get all transcripts for a session"""
    try:
        # Get from database
        db_transcripts = await db_manager.get_session_transcripts(session_id)
        
        # Get from memory (for recent ones)
        memory_transcripts = audio_service.transcripts.get(session_id, [])
        
        return {
            "session_id": session_id,
            "transcripts": db_transcripts,
            "recent_transcripts": memory_transcripts,
            "total_count": len(db_transcripts)
        }
    
    except Exception as e:
        logger.error(f"Error getting transcripts for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{session_id}")
async def get_processing_status(session_id: str):
    """Get processing status for a session"""
    try:
        # Get queue status
        session_queue = [item for item in audio_service.processing_queue if item["session_id"] == session_id]
        
        # Get database status
        audio_metadata = await db_manager.get_audio_chunks_metadata(session_id)
        
        status_counts = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
        for chunk in audio_metadata:
            status = chunk.get("processing_status", "pending")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "session_id": session_id,
            "queue_status": session_queue,
            "database_status": status_counts,
            "total_chunks": len(audio_metadata),
            "is_connected": session_id in audio_service.active_sessions
        }
    
    except Exception as e:
        logger.error(f"Error getting status for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-pending")
async def process_all_pending():
    """Process all pending audio chunks"""
    try:
        await process_pending_audio_chunks()
        return {"message": "Processing all pending audio chunks"}
    except Exception as e:
        logger.error(f"Error processing pending chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Audio Processing Service",
        "active_sessions": len(audio_service.active_sessions),
        "queue_size": len(audio_service.processing_queue),
        "timestamp": datetime.now().isoformat()
    }

# Simple HTML interface for testing
@app.get("/", response_class=HTMLResponse)
async def get_interface():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Processing Service</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .transcript { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
            .processing { background: #fff3cd; color: #856404; }
            button { background: #007bff; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            input[type="file"] { margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Audio Processing Service</h1>
            <p>Upload audio files or connect via WebSocket for real-time transcription.</p>
            
            <div>
                <h3>Upload Audio File</h3>
                <input type="file" id="audioFile" accept="audio/*">
                <button onclick="uploadAudio()">Process Audio</button>
            </div>
            
            <div>
                <h3>Session Status</h3>
                <input type="text" id="sessionId" placeholder="Enter session ID" value="test-session">
                <button onclick="getStatus()">Get Status</button>
                <button onclick="getTranscripts()">Get Transcripts</button>
            </div>
            
            <div id="status"></div>
            <div id="transcripts"></div>
        </div>

        <script>
            function uploadAudio() {
                const fileInput = document.getElementById('audioFile');
                const sessionId = document.getElementById('sessionId').value || 'test-session';
                
                if (!fileInput.files[0]) {
                    alert('Please select an audio file');
                    return;
                }
                
                const file = fileInput.files[0];
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    const base64Data = e.target.result.split(',')[1];
                    
                    fetch('/api/process-audio', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            audio_data: base64Data,
                            format: file.type.split('/')[1] || 'webm',
                            duration: 0,
                            session_id: sessionId
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').innerHTML = 
                            `<div class="status success">Audio uploaded successfully. Chunk ID: ${data.chunk_id}</div>`;
                    })
                    .catch(error => {
                        document.getElementById('status').innerHTML = 
                            `<div class="status error">Error: ${error.message}</div>`;
                    });
                };
                
                reader.readAsDataURL(file);
            }
            
            function getStatus() {
                const sessionId = document.getElementById('sessionId').value || 'test-session';
                
                fetch(`/api/status/${sessionId}`)
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').innerHTML = 
                            `<div class="status success">Status: ${JSON.stringify(data, null, 2)}</div>`;
                    })
                    .catch(error => {
                        document.getElementById('status').innerHTML = 
                            `<div class="status error">Error: ${error.message}</div>`;
                    });
            }
            
            function getTranscripts() {
                const sessionId = document.getElementById('sessionId').value || 'test-session';
                
                fetch(`/api/transcripts/${sessionId}`)
                    .then(response => response.json())
                    .then(data => {
                        let html = '<h3>Transcripts:</h3>';
                        data.transcripts.forEach((transcript, index) => {
                            html += `<div class="transcript">
                                <strong>Chunk ${index + 1}:</strong> ${transcript.transcript}<br>
                                <small>Confidence: ${transcript.transcript_confidence}, Time: ${transcript.created_at}</small>
                            </div>`;
                        });
                        document.getElementById('transcripts').innerHTML = html;
                    })
                    .catch(error => {
                        document.getElementById('transcripts').innerHTML = 
                            `<div class="status error">Error: ${error.message}</div>`;
                    });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    logger.info("Starting Audio Processing Service on port 8001...")
    uvicorn.run(
        "audio_service:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )