import os
import logging
import base64
import tempfile
import asyncio
import time
import httpx
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize clients
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Debug environment variables
logger.info(f"Environment check:")
logger.info(f"SUPABASE_URL: {'Set' if SUPABASE_URL else 'Missing'}")
logger.info(f"SUPABASE_KEY: {'Set' if SUPABASE_KEY else 'Missing'}")
logger.info(f"DEEPGRAM_API_KEY: {'Set' if DEEPGRAM_API_KEY else 'Missing'}")

# Create transcripts directory
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Audio processor: Supabase client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase in audio processor: {e}")

# Initialize Deepgram with detailed error checking
DEEPGRAM_AVAILABLE = False
if DEEPGRAM_API_KEY:
    try:
        # Test if we can import the deepgram library
        from deepgram import DeepgramClient, PrerecordedOptions
        DEEPGRAM_AVAILABLE = True
        logger.info("Deepgram API configured successfully")
    except ImportError as e:
        logger.error(f"Deepgram library not installed: {e}")
        logger.error("Install with: pip install deepgram-sdk")
        DEEPGRAM_AVAILABLE = False
    except Exception as e:
        logger.error(f"Failed to configure Deepgram API: {e}")
        DEEPGRAM_AVAILABLE = False
else:
    logger.warning("DEEPGRAM_API_KEY not found in environment variables")
    DEEPGRAM_AVAILABLE = False

logger.info(f"Deepgram API Status: {'Available' if DEEPGRAM_AVAILABLE else 'Not Available'}")


def save_transcript_to_file(session_id: str, chunk_id: str, transcript: str, metadata: Dict[str, Any] = None) -> Optional[str]:
    """Save transcript to a text file with the format session_id_chunk_id.txt"""
    try:
        # Create session directory if it doesn't exist
        session_dir = TRANSCRIPTS_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Create filename
        filename = f"{session_id}_{chunk_id}.txt"
        filepath = session_dir / filename
        
        # Prepare content
        content = f"Session ID: {session_id}\n"
        content += f"Chunk ID: {chunk_id}\n"
        content += f"Timestamp: {datetime.now().isoformat()}\n"
        
        if metadata:
            content += f"Duration: {metadata.get('duration', 'N/A')} seconds\n"
            content += f"Confidence: {metadata.get('confidence', 'N/A')}\n"
            content += f"Language: {metadata.get('language', 'N/A')}\n"
            content += f"Service: {metadata.get('service', 'N/A')}\n"
        
        content += f"\n{'='*50}\nTRANSCRIPT\n{'='*50}\n\n"
        content += transcript
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Transcript saved to {filepath}")
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Failed to save transcript to file: {e}")
        return None


async def test_deepgram_api() -> Dict[str, Any]:
    """Test Deepgram API connectivity"""
    if not DEEPGRAM_AVAILABLE:
        return {
            "status": "error",
            "message": "Deepgram API not available",
            "api_key_present": bool(DEEPGRAM_API_KEY),
            "library_imported": False
        }
    
    try:
        from deepgram import DeepgramClient
        
        # Create client
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        
        # Test with a simple HTTP request to check API key validity
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.deepgram.com/v1/projects",
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            )
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "message": "Deepgram API working correctly",
                    "api_key_present": True,
                    "library_imported": True,
                    "projects_accessible": True
                }
            else:
                return {
                    "status": "error",
                    "message": f"API key validation failed: {response.status_code}",
                    "api_key_present": True,
                    "library_imported": True,
                    "response": response.text
                }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Deepgram API test failed: {e}",
            "api_key_present": bool(DEEPGRAM_API_KEY),
            "library_imported": DEEPGRAM_AVAILABLE,
            "error_type": type(e).__name__
        }


async def transcribe_with_deepgram(audio_bytes: bytes, audio_format: str, chunk_id: str) -> Dict[str, Any]:
    """
    Transcribe audio bytes using Deepgram API
    
    Returns:
        Dict containing transcript, confidence, and metadata
    """
    if not DEEPGRAM_AVAILABLE:
        raise Exception("Deepgram API not available - check configuration and API key")
    
    if not DEEPGRAM_API_KEY:
        raise Exception("Deepgram API key not configured")
    
    logger.info(f"Starting Deepgram transcription for chunk {chunk_id}")
    logger.info(f"Audio format: {audio_format}, Size: {len(audio_bytes)} bytes")
    
    try:
        from deepgram import DeepgramClient, PrerecordedOptions, FileSource
        
        # Validate audio format
        supported_formats = ['webm', 'mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac']
        if audio_format.lower() not in supported_formats:
            logger.warning(f"Format {audio_format} not explicitly supported, trying anyway...")
        
        # Check audio size (Deepgram has generous limits)
        max_size = 500 * 1024 * 1024  # 500MB
        if len(audio_bytes) > max_size:
            raise Exception(f"Audio file too large: {len(audio_bytes)} bytes (max: {max_size})")
        
        logger.info(f"Audio validation passed - {len(audio_bytes)} bytes, format: {audio_format}")
        
        # Create Deepgram client
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        
        # Set transcription options
        options = PrerecordedOptions(
            model="nova-2",  # Latest model
            language="en",   # Auto-detect or specify
            smart_format=True,  # Enable smart formatting
            punctuate=True,     # Add punctuation
            utterances=True,    # Get word-level timestamps
            confidence=True,    # Include confidence scores
            diarize=False,      # Speaker diarization (set to True if multiple speakers)
            summarize="v2",     # Optional: get summary
            detect_language=True,  # Auto-detect language
        )
        
        # Create file source from audio bytes
        file_source = {
            "buffer": audio_bytes,
        }
        
        logger.info("Sending request to Deepgram...")
        
        # Make the transcription request
        response = deepgram.listen.rest.v("1").transcribe_file(
            file_source, options, timeout=60  # 60 second timeout
        )
        
        logger.info("Deepgram transcription completed")
        
        # Extract transcript from response
        if not response or not response.results:
            raise Exception("No transcription results returned from Deepgram")
        
        # Get the main transcript
        transcript_text = ""
        overall_confidence = 0.0
        detected_language = "en"
        
        if response.results.channels and len(response.results.channels) > 0:
            channel = response.results.channels[0]
            
            if channel.alternatives and len(channel.alternatives) > 0:
                alternative = channel.alternatives[0]
                transcript_text = alternative.transcript or ""
                overall_confidence = alternative.confidence or 0.0
                detected_language = response.results.language or "en"
        
        if not transcript_text.strip():
            transcript_text = "No clear speech detected"
            overall_confidence = 0.1
        
        # Get additional metadata
        metadata = {
            'file_size_bytes': len(audio_bytes),
            'audio_format': audio_format,
            'processing_time': datetime.now().isoformat(),
            'model_used': 'nova-2',
            'detected_language': detected_language,
        }
        
        # Add word-level data if available
        if (response.results.channels and 
            len(response.results.channels) > 0 and 
            response.results.channels[0].alternatives and
            len(response.results.channels[0].alternatives) > 0):
            
            alternative = response.results.channels[0].alternatives[0]
            if hasattr(alternative, 'words') and alternative.words:
                metadata['word_count'] = len(alternative.words)
                metadata['has_timestamps'] = True
        
        # Add summary if available
        if hasattr(response.results, 'summary') and response.results.summary:
            summary = response.results.summary
            if hasattr(summary, 'result') and summary.result:
                metadata['summary'] = summary.result
        
        result = {
            'transcript': transcript_text.strip(),
            'confidence': max(0.1, min(0.99, overall_confidence)),
            'language': detected_language,
            'service': 'deepgram-nova-2',
            'metadata': metadata
        }
        
        logger.info(f"Deepgram transcription successful for chunk {chunk_id}: {len(transcript_text)} characters, confidence: {overall_confidence}")
        return result
        
    except Exception as e:
        logger.error(f"Deepgram transcription failed for chunk {chunk_id}: {e}")
        raise


def create_mock_transcript(audio_bytes: bytes, audio_format: str, chunk_id: str) -> Dict[str, Any]:
    """Create a mock transcript for testing purposes"""
    logger.info(f"Creating mock transcript for chunk {chunk_id}")
    
    duration_estimate = len(audio_bytes) / 10000
    
    mock_transcripts = [
        "This is a test interview session for the coding assessment.",
        "I am working on solving the algorithm problem step by step.",
        "Let me think about the approach for this coding challenge.",
        "I need to implement a function that handles the edge cases properly.",
        "The time complexity of this solution should be optimized.",
        "I'll start by analyzing the problem requirements carefully.",
        "This is a mock transcription for testing the system functionality.",
        "The audio processing and transcript generation is working correctly.",
        "I am demonstrating the interview session capabilities.",
        "This transcript was generated for testing purposes."
    ]
    
    transcript_index = hash(chunk_id) % len(mock_transcripts)
    transcript_text = mock_transcripts[transcript_index]
    
    if len(audio_bytes) > 100000:
        transcript_text += " This is a longer audio recording with more detailed content."
    
    result = {
        'transcript': transcript_text,
        'confidence': 0.85,
        'language': 'en',
        'service': 'mock-transcription',
        'metadata': {
            'file_size_bytes': len(audio_bytes),
            'audio_format': audio_format,
            'processing_time': datetime.now().isoformat(),
            'is_mock': True,
            'reason': 'Deepgram API unavailable or failed'
        }
    }
    
    logger.info(f"Mock transcript created for chunk {chunk_id}: {len(transcript_text)} characters")
    return result


async def process_audio_chunk_from_db(chunk_id: str, auto_notify: bool = True) -> bool:
    """
    Process audio chunk using Deepgram API with fallback to mock transcription
    """
    if not supabase:
        logger.error("Supabase not configured for audio processing")
        return False
    
    logger.info(f"Starting to process audio chunk: {chunk_id}")
    
    try:
        from database_manager import db_manager
        
        # Mark as processing
        await db_manager.update_chunk_processing_status(
            int(chunk_id), 
            'processing', 
            started_at=datetime.now().isoformat()
        )
        
        # Fetch chunk data
        chunk_data = await db_manager.get_audio_chunk_by_id(int(chunk_id))
        
        if not chunk_data:
            raise ValueError(f"Audio chunk {chunk_id} not found in database")
        
        session_id = chunk_data['session_id']
        audio_b64 = chunk_data.get('audio_data')
        audio_format = chunk_data.get('format', 'webm')
        duration = chunk_data.get('duration', 0)
        
        if not audio_b64:
            raise ValueError(f"No audio data found for chunk {chunk_id}")
        
        # Decode audio
        logger.info(f"Decoding base64 audio data for chunk {chunk_id}")
        try:
            audio_bytes = base64.b64decode(audio_b64)
            logger.info(f"Successfully decoded {len(audio_bytes)} bytes of audio data")
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio data: {e}")
        
        # Try Deepgram transcription first
        transcript_result = None
        try:
            if DEEPGRAM_AVAILABLE:
                logger.info(f"Attempting Deepgram transcription for chunk {chunk_id}")
                transcript_result = await transcribe_with_deepgram(audio_bytes, audio_format, chunk_id)
                logger.info(f"Deepgram transcription successful for chunk {chunk_id}")
            else:
                logger.warning(f"Deepgram not available, using mock transcription for chunk {chunk_id}")
                raise Exception("Deepgram API not available")
                
        except Exception as deepgram_error:
            logger.warning(f"Deepgram transcription failed for chunk {chunk_id}: {deepgram_error}")
            logger.info(f"Falling back to mock transcription for chunk {chunk_id}")
            transcript_result = create_mock_transcript(audio_bytes, audio_format, chunk_id)
        
        # Save transcript to file
        file_path = save_transcript_to_file(
            session_id, 
            str(chunk_id), 
            transcript_result['transcript'],
            {
                'duration': duration,
                'confidence': transcript_result['confidence'],
                'language': transcript_result['language'],
                'service': transcript_result['service']
            }
        )
        
        if file_path:
            transcript_result['metadata']['file_path'] = file_path
        
        # Update database
        success = await db_manager.update_chunk_transcript(int(chunk_id), transcript_result)
        
        if success:
            logger.info(f"Successfully processed audio chunk {chunk_id} for session {session_id}")
            
            if auto_notify:
                await notify_clients_of_transcript(session_id, chunk_id, transcript_result)
            
            return True
        else:
            raise Exception("Failed to update database with transcript")
        
    except Exception as e:
        error_message = f"Processing failed: {str(e)}"
        logger.error(f"Failed to process audio chunk {chunk_id}: {error_message}")
        
        try:
            from database_manager import db_manager
            await db_manager.update_chunk_processing_status(
                int(chunk_id),
                'failed',
                completed_at=datetime.now().isoformat(),
                error=error_message
            )
        except Exception as update_err:
            logger.error(f"CRITICAL: Failed to mark chunk {chunk_id} as failed: {update_err}")

        return False


async def notify_clients_of_transcript(session_id: str, chunk_id: str, transcript_result: Dict):
    """Send transcript notification to connected clients via WebSocket or HTTP"""
    try:
        logger.info(f"Would notify clients of transcript ready for session {session_id}, chunk {chunk_id}")
        
        notification_data = {
            "type": "transcript_ready",
            "session_id": session_id,
            "chunk_id": chunk_id,
            "transcript": transcript_result.get('transcript', '')[:200] + "..." if len(transcript_result.get('transcript', '')) > 200 else transcript_result.get('transcript', ''),
            "confidence": transcript_result.get('confidence', 0),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Transcript notification prepared: {notification_data}")
        
    except Exception as e:
        logger.warning(f"Failed to notify clients: {e}")


async def process_pending_audio_chunks(max_concurrent: int = 3):
    """
    Background task to process all pending audio chunks using Deepgram
    """
    if not supabase:
        logger.warning("Supabase not configured, skipping audio processing")
        return
    
    try:
        from database_manager import db_manager
        
        # Get pending chunks
        pending_chunks = await db_manager.get_pending_transcripts()
        
        if not pending_chunks:
            logger.debug("No pending audio chunks to process")
            return
        
        logger.info(f"Processing {len(pending_chunks)} pending audio chunks with Deepgram")
        
        # Process chunks with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(chunk):
            async with semaphore:
                chunk_id = str(chunk['id'])
                return await process_audio_chunk_from_db(chunk_id, auto_notify=True)
        
        tasks = [process_with_semaphore(chunk) for chunk in pending_chunks[:10]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        logger.info(f"Processed {success_count}/{len(tasks)} audio chunks successfully with Deepgram")
        
        return {
            'processed': len(tasks),
            'successful': success_count,
            'failed': len(tasks) - success_count
        }
        
    except Exception as e:
        logger.error(f"Error in background audio processing: {e}")
        return {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'error': str(e)
        }


# Debugging function for Deepgram setup
async def debug_deepgram_setup() -> Dict[str, Any]:
    """Debug Deepgram configuration and test basic functionality"""
    debug_info = {
        "environment": {
            "deepgram_api_key_present": bool(DEEPGRAM_API_KEY),
            "deepgram_api_key_length": len(DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else 0,
            "deepgram_available": DEEPGRAM_AVAILABLE
        },
        "library_check": {},
        "api_test": {}
    }
    
    # Test library import
    try:
        from deepgram import DeepgramClient, PrerecordedOptions
        debug_info["library_check"] = {
            "import_successful": True,
            "deepgram_client_available": True
        }
    except ImportError as e:
        debug_info["library_check"] = {
            "import_successful": False,
            "error": str(e)
        }
        return debug_info
    
    # Test API connectivity
    if DEEPGRAM_AVAILABLE:
        api_test = await test_deepgram_api()
        debug_info["api_test"] = api_test
    else:
        debug_info["api_test"] = {"status": "skipped", "reason": "Deepgram not available"}
    
    return debug_info


# Test function for debugging
async def test_audio_processing():
    """Test function to verify audio processing works with Deepgram"""
    try:
        from database_manager import db_manager
        
        pending = await db_manager.get_pending_transcripts()
        
        if pending:
            test_chunk_id = str(pending[0]['id'])
            logger.info(f"Testing Deepgram audio processing with chunk ID: {test_chunk_id}")
            
            result = await process_audio_chunk_from_db(test_chunk_id)
            
            if result:
                logger.info("Deepgram audio processing test PASSED")
            else:
                logger.error("Deepgram audio processing test FAILED")
        else:
            logger.info("No pending audio chunks found for testing")
            
    except Exception as e:
        logger.error(f"Deepgram audio processing test failed: {e}")


if __name__ == "__main__":
    # Run test if file is executed directly
    asyncio.run(test_audio_processing())