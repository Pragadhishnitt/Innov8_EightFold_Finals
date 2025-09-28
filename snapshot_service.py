import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from database_manager import db_manager, supabase

logger = logging.getLogger(__name__)

class SnapshotService:
    """Service for handling sandbox snapshots at regular intervals"""
    
    def __init__(self):
        self.snapshots_dir = Path("snapshots")
        self.snapshots_dir.mkdir(exist_ok=True)
        self.latest_snapshot = None
        self.snapshot_interval = 60  # 1 minute in seconds
        self.is_running = False
        logger.info("Snapshot service initialized")
    
    async def start_background_snapshot_task(self):
        """Start the background task for periodic snapshots"""
        if self.is_running:
            logger.warning("Snapshot task is already running")
            return
        
        self.is_running = True
        logger.info(f"Starting background snapshot task with {self.snapshot_interval}s interval")
        
        try:
            while self.is_running:
                await self.take_snapshot()
                await asyncio.sleep(self.snapshot_interval)
        except asyncio.CancelledError:
            logger.info("Snapshot task cancelled")
        except Exception as e:
            logger.error(f"Error in snapshot task: {e}")
        finally:
            self.is_running = False
    
    def stop_background_snapshot_task(self):
        """Stop the background snapshot task"""
        self.is_running = False
        logger.info("Stopping background snapshot task")
    
    async def take_snapshot(self) -> Dict[str, Any]:
        """
        Take a snapshot of the current sandbox environment.
        
        Returns:
            Dict containing snapshot data and metadata
        """
        try:
            logger.info("Taking sandbox snapshot")
            
            # Get all active sessions data
            sessions_data = await self._get_sessions_data()
            
            # Get audio chunks data
            audio_data = await self._get_audio_data()
            
            # Get code history data
            code_data = await self._get_code_data()
            
            # Get execution results data
            execution_data = await self._get_execution_data()
            
            # Create snapshot
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "snapshot_id": f"snapshot_{int(datetime.now().timestamp())}",
                "data": {
                    "sessions": sessions_data,
                    "audio_chunks": audio_data,
                    "code_history": code_data,
                    "execution_results": execution_data
                },
                "metadata": {
                    "total_sessions": len(sessions_data),
                    "total_audio_chunks": len(audio_data),
                    "total_code_entries": len(code_data),
                    "total_executions": len(execution_data),
                    "snapshot_size_bytes": 0  # Will be calculated after saving
                }
            }
            
            # Save snapshot to file
            file_path = await self._save_snapshot_to_file(snapshot)
            
            # Update metadata with file size
            if file_path and os.path.exists(file_path):
                snapshot["metadata"]["snapshot_size_bytes"] = os.path.getsize(file_path)
                snapshot["file_path"] = str(file_path)
            
            # Store as latest snapshot
            self.latest_snapshot = snapshot
            
            logger.info(f"Snapshot taken successfully: {snapshot['snapshot_id']}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Error taking snapshot: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_sessions_data(self) -> List[Dict[str, Any]]:
        """Get all sessions data from database"""
        try:
            if not supabase:
                return []
            
            response = supabase.table('sessions').select('*').execute()
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error getting sessions data: {e}")
            return []
    
    async def _get_audio_data(self) -> List[Dict[str, Any]]:
        """Get all audio chunks data from database"""
        try:
            if not supabase:
                return []
            
            response = supabase.table('audio_chunks').select(
                'id, session_id, format, duration, size_kb, processing_status, '
                'transcript, transcript_confidence, transcript_language, '
                'transcript_service, created_at, processing_started_at, processing_completed_at'
            ).execute()
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error getting audio data: {e}")
            return []
    
    async def _get_code_data(self) -> List[Dict[str, Any]]:
        """Get all code history data from database"""
        try:
            if not supabase:
                return []
            
            response = supabase.table('code_history').select(
                'id, session_id, code, cursor_position, line_count, char_count, '
                'code_hash, created_at'
            ).execute()
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error getting code data: {e}")
            return []
    
    async def _get_execution_data(self) -> List[Dict[str, Any]]:
        """Get all execution results data from database"""
        try:
            if not supabase:
                return []
            
            response = supabase.table('execution_results').select(
                'id, session_id, code, code_hash, success, output, error, '
                'execution_time, security_level, language, created_at'
            ).execute()
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error getting execution data: {e}")
            return []
    
    async def _save_snapshot_to_file(self, snapshot: Dict[str, Any]) -> Optional[str]:
        """Save snapshot to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.json"
            file_path = self.snapshots_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Snapshot saved to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving snapshot to file: {e}")
            return None
    
    async def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot"""
        return self.latest_snapshot
    
    async def get_snapshot_by_id(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific snapshot by ID"""
        try:
            # First check if it's the latest snapshot
            if self.latest_snapshot and self.latest_snapshot.get('snapshot_id') == snapshot_id:
                return self.latest_snapshot
            
            # Search in files
            for file_path in self.snapshots_dir.glob("snapshot_*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        snapshot = json.load(f)
                        if snapshot.get('snapshot_id') == snapshot_id:
                            return snapshot
                except Exception as e:
                    logger.warning(f"Error reading snapshot file {file_path}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting snapshot by ID {snapshot_id}: {e}")
            return None
    
    async def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots"""
        try:
            snapshots = []
            
            # Add latest snapshot if available
            if self.latest_snapshot:
                snapshots.append({
                    "snapshot_id": self.latest_snapshot.get('snapshot_id'),
                    "timestamp": self.latest_snapshot.get('timestamp'),
                    "file_path": self.latest_snapshot.get('file_path'),
                    "is_latest": True
                })
            
            # Add file-based snapshots
            for file_path in sorted(self.snapshots_dir.glob("snapshot_*.json"), reverse=True):
                try:
                    stat = file_path.stat()
                    snapshots.append({
                        "snapshot_id": file_path.stem,
                        "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "file_path": str(file_path),
                        "file_size": stat.st_size,
                        "is_latest": False
                    })
                except Exception as e:
                    logger.warning(f"Error reading snapshot file {file_path}: {e}")
                    continue
            
            return snapshots
            
        except Exception as e:
            logger.error(f"Error listing snapshots: {e}")
            return []
    
    async def cleanup_old_snapshots(self, keep_days: int = 7):
        """Clean up old snapshot files"""
        try:
            cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)
            deleted_count = 0
            
            for file_path in self.snapshots_dir.glob("snapshot_*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old snapshot: {file_path}")
                    except Exception as e:
                        logger.warning(f"Error deleting snapshot {file_path}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old snapshots")
            
        except Exception as e:
            logger.error(f"Error cleaning up snapshots: {e}")

# Create singleton instance
snapshot_service = SnapshotService()
