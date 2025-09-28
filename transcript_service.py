import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from database_manager import db_manager
from audio_processor import process_audio_chunk_from_db, save_transcript_to_file

logger = logging.getLogger(__name__)

class TranscriptService:
    """Service for handling transcript generation and file management"""
    
    def __init__(self):
        self.transcripts_dir = Path("transcripts")
        self.transcripts_dir.mkdir(exist_ok=True)
        logger.info("Transcript service initialized")
    
    async def generate_transcript_for_chunk(self, chunk_id: str, session_id: str) -> Dict[str, Any]:
        """
        Generate transcript for a specific audio chunk and save to file.
        
        Args:
            chunk_id: The ID of the audio chunk
            session_id: The session ID
            
        Returns:
            Dict containing transcript information and file path
        """
        try:
            logger.info(f"Starting transcript generation for chunk {chunk_id} in session {session_id}")
            
            # Process the audio chunk using existing functionality
            success = await process_audio_chunk_from_db(chunk_id, auto_notify=False)
            
            if not success:
                raise Exception(f"Failed to process audio chunk {chunk_id}")
            
            # Get the transcript data from database
            transcripts = await db_manager.get_session_transcripts(session_id)
            latest_transcript = None
            
            # Find the transcript for this specific chunk
            for transcript in transcripts:
                if str(transcript.get('id')) == str(chunk_id):
                    latest_transcript = transcript
                    break
            
            if not latest_transcript:
                raise Exception(f"No transcript found for chunk {chunk_id}")
            
            # Generate filename
            filename = f"{session_id}_{chunk_id}.txt"
            file_path = self.transcripts_dir / session_id / filename
            
            # Ensure session directory exists
            file_path.parent.mkdir(exist_ok=True)
            
            # Create transcript content
            transcript_content = self._format_transcript_content(
                session_id, chunk_id, latest_transcript
            )
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(transcript_content)
            
            logger.info(f"Transcript saved to {file_path}")
            
            return {
                "success": True,
                "chunk_id": chunk_id,
                "session_id": session_id,
                "file_path": str(file_path),
                "filename": filename,
                "transcript": latest_transcript.get('transcript', ''),
                "confidence": latest_transcript.get('transcript_confidence', 0.0),
                "language": latest_transcript.get('transcript_language', 'en'),
                "duration": latest_transcript.get('duration', 0.0),
                "created_at": latest_transcript.get('created_at', ''),
                "file_size": file_path.stat().st_size
            }
            
        except Exception as e:
            logger.error(f"Error generating transcript for chunk {chunk_id}: {e}")
            return {
                "success": False,
                "chunk_id": chunk_id,
                "session_id": session_id,
                "error": str(e),
                "file_path": None
            }
    
    def _format_transcript_content(self, session_id: str, chunk_id: str, transcript_data: Dict[str, Any]) -> str:
        """Format transcript content for file storage"""
        content = f"Session ID: {session_id}\n"
        content += f"Chunk ID: {chunk_id}\n"
        content += f"Generated: {datetime.now().isoformat()}\n"
        content += f"Duration: {transcript_data.get('duration', 'N/A')} seconds\n"
        content += f"Confidence: {transcript_data.get('transcript_confidence', 'N/A')}\n"
        content += f"Language: {transcript_data.get('transcript_language', 'N/A')}\n"
        content += f"Service: {transcript_data.get('transcript_service', 'N/A')}\n"
        content += f"Created: {transcript_data.get('created_at', 'N/A')}\n"
        content += f"\n{'='*50}\nTRANSCRIPT\n{'='*50}\n\n"
        content += transcript_data.get('transcript', 'No transcript available')
        
        return content
    
    async def get_transcript_file(self, session_id: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get transcript file information if it exists"""
        try:
            filename = f"{session_id}_{chunk_id}.txt"
            file_path = self.transcripts_dir / session_id / filename
            
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            return {
                "filename": filename,
                "file_path": str(file_path),
                "file_size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "exists": True
            }
            
        except Exception as e:
            logger.error(f"Error getting transcript file for {session_id}_{chunk_id}: {e}")
            return None
    
    async def list_session_transcripts(self, session_id: str) -> list:
        """List all transcript files for a session"""
        try:
            session_dir = self.transcripts_dir / session_id
            if not session_dir.exists():
                return []
            
            transcript_files = []
            for file_path in session_dir.glob("*.txt"):
                stat = file_path.stat()
                transcript_files.append({
                    "filename": file_path.name,
                    "file_path": str(file_path),
                    "file_size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            
            return sorted(transcript_files, key=lambda x: x['filename'])
            
        except Exception as e:
            logger.error(f"Error listing transcripts for session {session_id}: {e}")
            return []

# Create singleton instance
transcript_service = TranscriptService()
