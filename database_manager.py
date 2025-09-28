import os
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from supabase import create_client, Client
from dotenv import load_dotenv
import asyncio
import json

# Basic logger setup for this module
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Database manager initialized Supabase client successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client in database_manager: {e}")
else:
    logger.warning("Supabase credentials not found. Database features will be disabled.")


class DatabaseManager:
    """Handles all interactions with the Supabase database with enhanced audio and transcript support."""

    @staticmethod
    async def ensure_session_exists(session_id: str) -> bool:
        """Ensure session exists in database, create if it doesn't"""
        if not supabase:
            return False
            
        try:
            # First check if session exists
            response = supabase.table('sessions').select('session_id').eq('session_id', session_id).execute()
            
            if response.data:
                # Session exists, update last_activity
                supabase.table('sessions').update({
                    'last_activity': datetime.now().isoformat(),
                    'is_active': True
                }).eq('session_id', session_id).execute()
                return True
            else:
                # Session doesn't exist, create it
                response = supabase.table('sessions').insert({
                    'session_id': session_id,
                    'is_active': True,
                    'created_at': datetime.now().isoformat(),
                    'last_activity': datetime.now().isoformat()
                }).execute()
                
                if response.data:
                    logger.info(f"Created new session: {session_id}")
                    return True
                else:
                    logger.error(f"Failed to create session: {session_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error ensuring session exists for {session_id}: {e}")
            return False

    @staticmethod
    async def store_session(session_id: str) -> bool:
        """Store/update session - now calls ensure_session_exists"""
        return await DatabaseManager.ensure_session_exists(session_id)

    @staticmethod
    async def update_session_activity(session_id: str, is_active: bool = True):
        if not supabase: return
        try:
            supabase.table('sessions').update({
                'last_activity': datetime.now().isoformat(),
                'is_active': is_active
            }).eq('session_id', session_id).execute()
        except Exception as e:
            logger.error(f"Failed to update activity for {session_id}: {e}")

    @staticmethod
    async def store_code_update(session_id: str, code_entry: dict) -> bool:
        """Store code update - ensures session exists first"""
        if not supabase:
            return False
            
        try:
            # Ensure session exists
            if not await DatabaseManager.ensure_session_exists(session_id):
                raise Exception(f"Cannot create session {session_id}")
            
            supabase.table('code_history').insert({
                'session_id': session_id,
                'code': code_entry['code'],
                'cursor_position': code_entry['cursor_position'],
                'line_count': code_entry['line_count'],
                'char_count': code_entry['char_count'],
                'code_hash': code_entry.get('hash', ''),
                'created_at': code_entry.get('timestamp', datetime.now().isoformat())
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to store code update for {session_id}: {e}")
            return False

    @staticmethod
    async def store_execution_result(session_id: str, result_entry: dict) -> bool:
        """Store execution result - ensures session exists first"""
        if not supabase:
            return False
            
        try:
            # Ensure session exists
            if not await DatabaseManager.ensure_session_exists(session_id):
                raise Exception(f"Cannot create session {session_id}")
                
            result = result_entry['result']
            supabase.table('execution_results').insert({
                'session_id': session_id,
                'code': result_entry['code'],
                'code_hash': result_entry.get('code_hash', ''),
                'success': result['success'],
                'output': result.get('output', ''),
                'error': result.get('error', ''),
                'execution_time': result.get('execution_time', 'N/A'),
                'security_level': result.get('security_level', 'Unknown'),
                'language': result_entry.get('language', 'python'),
                'memory_usage': result.get('memory_usage', 'N/A'),
                'created_at': result_entry.get('timestamp', datetime.now().isoformat())
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to store execution result for {session_id}: {e}")
            return False

# Enhanced debugging version for database_manager.py

    @staticmethod
    async def store_audio_chunk(session_id: str, audio_entry: dict) -> Optional[int]:
        """ENHANCED DEBUG VERSION: Store audio chunk with comprehensive logging"""
        if not supabase: 
            logger.error("❌ Supabase client not available")
            return None
        
        try:
            logger.info(f"🔄 Starting audio chunk storage for session: {session_id}")
            
            # Step 1: Ensure session exists
            session_exists = await DatabaseManager.ensure_session_exists(session_id)
            logger.info(f"Session exists check: {session_exists}")
            
            if not session_exists:
                raise Exception(f"Failed to ensure session exists: {session_id}")
            
            # Step 2: Validate audio data
            audio_data = audio_entry.get('audio_data', '')
            if not audio_data:
                raise Exception("No audio_data in audio_entry")
            
            logger.info(f"Audio data length: {len(audio_data)} characters")
            
            # Step 3: Prepare insert data
            size_bytes = len(audio_data.encode('utf-8'))
            size_kb = round(size_bytes / 1024, 2)
            
            insert_data = {
                'session_id': session_id,
                'audio_data': audio_data,
                'format': audio_entry.get('format', 'webm'),
                'duration': float(audio_entry.get('duration', 0)),
                'size_kb': size_kb,
                'size_bytes': size_bytes,
                'audio_hash': audio_entry.get('audio_hash', ''),
                'chunk_index': audio_entry.get('chunk_index'),
                'processing_status': 'pending',
                'transcript': None,
                'transcript_confidence': None,
                'transcript_language': None,
                'transcript_service': None,
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"📊 Insert data prepared:")
            logger.info(f"  - Size: {size_kb}KB ({size_bytes} bytes)")
            logger.info(f"  - Format: {insert_data['format']}")
            logger.info(f"  - Duration: {insert_data['duration']}s")
            logger.info(f"  - Session: {session_id}")
            
            # Step 4: Test basic database connectivity first
            logger.info("🔍 Testing database connectivity...")
            test_response = supabase.table('audio_chunks').select('id').limit(1).execute()
            logger.info(f"Database test response: {test_response}")
            
            # Step 5: Execute the actual insert
            logger.info("💾 Executing database insert...")
            
            # Try the insert with detailed error catching
            try:
                response = supabase.table('audio_chunks').insert(insert_data).execute()
                logger.info(f"✅ Insert completed")
                logger.info(f"Response type: {type(response)}")
                logger.info(f"Response dir: {dir(response)}")
                logger.info(f"Full response: {response}")
                
                # Check if response has data attribute
                if hasattr(response, 'data'):
                    logger.info(f"Response.data: {response.data}")
                    logger.info(f"Response.data type: {type(response.data)}")
                    
                    if response.data is None:
                        logger.error("❌ response.data is None")
                        
                        # Check for errors
                        if hasattr(response, 'error') and response.error:
                            logger.error(f"Database error: {response.error}")
                        
                        return None
                    
                    if isinstance(response.data, list) and len(response.data) > 0:
                        record = response.data[0]
                        logger.info(f"First record: {record}")
                        logger.info(f"Record keys: {list(record.keys()) if isinstance(record, dict) else 'Not a dict'}")
                        
                        chunk_id = record.get('id') if isinstance(record, dict) else None
                        logger.info(f"Extracted chunk_id: {chunk_id} (type: {type(chunk_id)})")
                        
                        if chunk_id is not None:
                            return int(chunk_id)
                        else:
                            logger.error("❌ No 'id' field in record")
                            return None
                    else:
                        logger.error(f"❌ Invalid data structure: {response.data}")
                        return None
                else:
                    logger.error("❌ Response has no 'data' attribute")
                    return None
                    
            except Exception as insert_error:
                logger.error(f"❌ Insert operation failed: {insert_error}")
                logger.error(f"Insert error type: {type(insert_error)}")
                
                # Try to get more details about the error
                if hasattr(insert_error, 'details'):
                    logger.error(f"Error details: {insert_error.details}")
                if hasattr(insert_error, 'message'):
                    logger.error(f"Error message: {insert_error.message}")
                if hasattr(insert_error, 'code'):
                    logger.error(f"Error code: {insert_error.code}")
                
                raise insert_error
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Audio chunk storage failed for {session_id}: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Exception args: {e.args}")
            
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            return None

    # Alternative method to test direct database operations
    @staticmethod
    async def test_audio_chunk_insert() -> dict:
        """Test method to verify database insert functionality"""
        if not supabase:
            return {"status": "error", "message": "Supabase not configured"}
        
        try:
            # Test session creation first
            test_session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create test session
            session_response = supabase.table('sessions').insert({
                'session_id': test_session_id,
                'is_active': True,
                'created_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat()
            }).execute()
            
            logger.info(f"Test session created: {session_response}")
            
            # Test audio chunk insertion
            test_audio_data = {
                'session_id': test_session_id,
                'audio_data': 'dGVzdF9hdWRpb19kYXRh',  # base64 for "test_audio_data"
                'format': 'webm',
                'duration': 5.0,
                'size_kb': 1.0,
                'size_bytes': 1024,
                'audio_hash': 'test_hash',
                'chunk_index': 0,
                'processing_status': 'pending',
                'created_at': datetime.now().isoformat()
            }
            
            audio_response = supabase.table('audio_chunks').insert(test_audio_data).execute()
            logger.info(f"Test audio chunk created: {audio_response}")
            
            # Clean up test data
            supabase.table('audio_chunks').delete().eq('session_id', test_session_id).execute()
            supabase.table('sessions').delete().eq('session_id', test_session_id).execute()
            
            return {
                "status": "success",
                "session_response": str(session_response),
                "audio_response": str(audio_response),
                "message": "Database insert test completed successfully"
            }
            
        except Exception as e:
            logger.error(f"Database test failed: {e}")
            return {
                "status": "error", 
                "message": f"Database test failed: {e}",
                "error_type": type(e).__name__
            }

    # Method to check database schema
    @staticmethod
    async def check_database_schema() -> dict:
        """Check if the database tables have the expected structure"""
        if not supabase:
            return {"status": "error", "message": "Supabase not configured"}
        
        try:
            # Try to select from both tables to check they exist
            sessions_test = supabase.table('sessions').select('*').limit(1).execute()
            audio_chunks_test = supabase.table('audio_chunks').select('*').limit(1).execute()
            
            return {
                "status": "success",
                "sessions_table": "exists" if sessions_test else "missing",
                "audio_chunks_table": "exists" if audio_chunks_test else "missing",
                "message": "Schema check completed"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Schema check failed: {e}",
                "error_type": type(e).__name__
            }
    @staticmethod
    async def get_audio_chunk_by_id(chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get audio chunk by ID - NEEDED FOR PROCESSING"""
        if not supabase:
            return None
        
        try:
            response = supabase.table('audio_chunks').select('*').eq('id', chunk_id).single().execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get audio chunk {chunk_id}: {e}")
            return None

    @staticmethod
    async def update_chunk_processing_status(chunk_id: int, status: str, started_at: str = None, completed_at: str = None, error: str = None):
        """Update processing status of audio chunk"""
        if not supabase:
            return False
        
        try:
            update_data = {'processing_status': status}
            
            if started_at:
                update_data['processing_started_at'] = started_at
            if completed_at:
                update_data['processing_completed_at'] = completed_at
            if error:
                update_data['processing_error'] = error
                
            supabase.table('audio_chunks').update(update_data).eq('id', chunk_id).execute()
            logger.info(f"Updated chunk {chunk_id} status to {status}")
            return True
        except Exception as e:
            logger.error(f"Failed to update chunk {chunk_id} status: {e}")
            return False

    @staticmethod
    async def update_chunk_transcript(chunk_id: int, transcript_result: Dict[str, Any]) -> bool:
        """Update chunk with transcript data"""
        if not supabase:
            return False
        
        try:
            update_data = {
                'processing_status': 'completed',
                'transcript': transcript_result.get('transcript', ''),
                'transcript_confidence': float(transcript_result.get('confidence', 0.0)),
                'transcript_language': transcript_result.get('language', 'en'),
                'transcript_service': transcript_result.get('service', 'gemini'),
                'processing_completed_at': datetime.now().isoformat()
            }
            
            # Store metadata as JSON if present
            if 'metadata' in transcript_result:
                update_data['transcript_metadata'] = json.dumps(transcript_result['metadata'])
            
            supabase.table('audio_chunks').update(update_data).eq('id', chunk_id).execute()
            logger.info(f"Updated transcript for chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update transcript for chunk {chunk_id}: {e}")
            return False

    # Enhanced audio retrieval methods for the API endpoints
    @staticmethod
    async def get_audio_chunk(session_id: str, chunk_index: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific audio chunk by session and index"""
        if not supabase: 
            return None
        
        try:
            response = supabase.table('audio_chunks').select('*').eq('session_id', session_id).order('created_at').execute()
            
            if response.data and len(response.data) > chunk_index:
                return response.data[chunk_index]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve audio chunk {chunk_index} for session {session_id}: {e}")
            return None

    @staticmethod
    async def get_audio_chunk_metadata(session_id: str, chunk_index: int) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific audio chunk without the audio data"""
        if not supabase:
            return None
        
        try:
            response = supabase.table('audio_chunks').select(
                'id, session_id, format, duration, size_kb, size_bytes, audio_hash, '
                'processing_status, transcript, transcript_confidence, transcript_language, '
                'transcript_service, created_at, processing_started_at, processing_completed_at'
            ).eq('session_id', session_id).order('created_at').execute()
            
            if response.data and len(response.data) > chunk_index:
                return response.data[chunk_index]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve audio metadata {chunk_index} for session {session_id}: {e}")
            return None

    @staticmethod
    async def get_all_audio_chunks(session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all audio chunks for a session"""
        if not supabase:
            return []
        
        try:
            response = supabase.table('audio_chunks').select('*').eq('session_id', session_id).order('created_at').execute()
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to retrieve all audio chunks for session {session_id}: {e}")
            return []

    @staticmethod
    async def get_audio_chunks_metadata(session_id: str) -> List[Dict[str, Any]]:
        """Retrieve metadata for all audio chunks without the audio data"""
        if not supabase:
            return []
        
        try:
            response = supabase.table('audio_chunks').select(
                'id, session_id, format, duration, size_kb, size_bytes, audio_hash, '
                'processing_status, transcript, transcript_confidence, transcript_language, '
                'transcript_service, created_at, processing_started_at, processing_completed_at'
            ).eq('session_id', session_id).order('created_at').execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to retrieve audio metadata for session {session_id}: {e}")
            return []

    @staticmethod
    async def get_pending_transcripts() -> List[Dict[str, Any]]:
        """Get all audio chunks pending transcript processing"""
        if not supabase:
            return []
        
        try:
            response = supabase.table('audio_chunks').select(
                'id, session_id, audio_data, format, duration'
            ).eq('processing_status', 'pending').order('created_at').execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to retrieve pending transcripts: {e}")
            return []

    @staticmethod
    async def get_session_transcripts(session_id: str) -> List[Dict[str, Any]]:
        """Get all transcripts for a session in chronological order"""
        if not supabase:
            return []
        
        try:
            response = supabase.table('audio_chunks').select(
                'id, created_at, transcript, transcript_confidence, transcript_language, '
                'transcript_service, duration, processing_status'
            ).eq('session_id', session_id).not_.is_('transcript', 'null').order('created_at').execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error(f"Failed to retrieve transcripts for session {session_id}: {e}")
            return []

    @staticmethod
    async def get_full_session_transcript(session_id: str) -> str:
        """Get combined transcript text for entire session"""
        transcripts = await DatabaseManager.get_session_transcripts(session_id)
        
        if not transcripts:
            return ""
        
        # Combine all transcripts with timestamps
        full_transcript = []
        for transcript in transcripts:
            if transcript.get('transcript'):
                timestamp = transcript.get('created_at', '')
                text = transcript.get('transcript', '')
                confidence = transcript.get('transcript_confidence', 0)
                
                full_transcript.append(f"[{timestamp[:19]}] ({confidence:.1%} confidence) {text}")
        
        return "\n".join(full_transcript)

    @staticmethod
    async def get_session_analytics(session_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a session including transcripts"""
        if not supabase:
            return {}
        
        try:
            # Get session info
            session_response = supabase.table('sessions').select('*').eq('session_id', session_id).execute()
            
            # Get audio analytics
            audio_response = supabase.table('audio_chunks').select(
                'duration, processing_status, transcript_confidence, transcript_language'
            ).eq('session_id', session_id).execute()
            
            # Calculate metrics
            audio_data = audio_response.data or []
            total_duration = sum(chunk.get('duration', 0) for chunk in audio_data)
            processed_chunks = len([chunk for chunk in audio_data if chunk.get('processing_status') == 'completed'])
            avg_confidence = 0
            
            if processed_chunks > 0:
                confidences = [chunk.get('transcript_confidence', 0) for chunk in audio_data 
                                if chunk.get('transcript_confidence') is not None]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
            
            return {
                'session_id': session_id,
                'session_info': session_response.data[0] if session_response.data else {},
                'audio_metrics': {
                    'total_chunks': len(audio_data),
                    'total_duration_seconds': total_duration,
                    'processed_chunks': processed_chunks,
                    'pending_chunks': len([chunk for chunk in audio_data if chunk.get('processing_status') == 'pending']),
                    'failed_chunks': len([chunk for chunk in audio_data if chunk.get('processing_status') == 'failed']),
                    'average_transcript_confidence': avg_confidence,
                    'languages_detected': list(set(chunk.get('transcript_language') for chunk in audio_data if chunk.get('transcript_language')))
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get session analytics for {session_id}: {e}")
            return {}

    @staticmethod
    async def cleanup_old_audio_data(days_old: int = 30):
        """Clean up old audio data (keeping transcripts)"""
        if not supabase:
            return
        
        try:
            cutoff_date = datetime.now().replace(day=datetime.now().day - days_old).isoformat()
            
            # Clear audio_data but keep transcripts and metadata
            supabase.table('audio_chunks').update({
                'audio_data': None,
                'cleanup_date': datetime.now().isoformat()
            }).lt('created_at', cutoff_date).execute()
            
            logger.info(f"Cleaned up audio data older than {days_old} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old audio data: {e}")


# Create a single instance that can be imported by other files
db_manager = DatabaseManager()