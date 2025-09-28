# FastAPI Interview Session Operations - API Documentation

This document describes the new endpoints added to support transcript creation and sandbox snapshots for interview sessions.

## Overview

Two new operations have been added to the FastAPI project:

1. **Transcript Creation on Audio Chunk Insert** - Generate transcript files when audio chunks are processed
2. **Sandbox Snapshot at Regular Intervals** - Capture and store sandbox environment data periodically

## New Endpoints

### 1. Transcript Generation Endpoints

#### `POST /process_chunk/{chunk_id}`
Generate a transcript for a specific audio chunk.

**Parameters:**
- `chunk_id` (path): The ID of the audio chunk to process
- `session_id` (query, optional): Session ID (will be fetched from database if not provided)

**Response:**
```json
{
  "success": true,
  "message": "Transcript generated successfully",
  "data": {
    "success": true,
    "chunk_id": "123",
    "session_id": "session-456",
    "file_path": "/path/to/transcript/file.txt",
    "filename": "session-456_123.txt",
    "transcript": "Transcribed text content...",
    "confidence": 0.95,
    "language": "en",
    "duration": 5.2,
    "created_at": "2024-01-01T12:00:00Z",
    "file_size": 1024
  }
}
```

#### `GET /transcript/{session_id}/{chunk_id}`
Get transcript file information for a specific chunk.

**Parameters:**
- `session_id` (path): The session ID
- `chunk_id` (path): The chunk ID

**Response:**
```json
{
  "success": true,
  "message": "Transcript file found",
  "data": {
    "filename": "session-456_123.txt",
    "file_path": "/path/to/transcript/file.txt",
    "file_size": 1024,
    "created_at": "2024-01-01T12:00:00Z",
    "modified_at": "2024-01-01T12:00:00Z",
    "exists": true
  }
}
```

#### `GET /transcripts/{session_id}`
List all transcript files for a session.

**Parameters:**
- `session_id` (path): The session ID

**Response:**
```json
{
  "success": true,
  "message": "Found 3 transcript files",
  "data": {
    "session_id": "session-456",
    "transcripts": [
      {
        "filename": "session-456_123.txt",
        "file_path": "/path/to/transcript/file1.txt",
        "file_size": 1024,
        "created_at": "2024-01-01T12:00:00Z",
        "modified_at": "2024-01-01T12:00:00Z"
      }
    ],
    "total_count": 3
  }
}
```

### 2. Sandbox Snapshot Endpoints

#### `GET /sandbox_snapshot`
Get the most recent sandbox snapshot.

**Response:**
```json
{
  "success": true,
  "message": "Latest snapshot retrieved successfully",
  "data": {
    "timestamp": "2024-01-01T12:00:00Z",
    "snapshot_id": "snapshot_1704110400",
    "data": {
      "sessions": [...],
      "audio_chunks": [...],
      "code_history": [...],
      "execution_results": [...]
    },
    "metadata": {
      "total_sessions": 5,
      "total_audio_chunks": 25,
      "total_code_entries": 100,
      "total_executions": 50,
      "snapshot_size_bytes": 2048000
    },
    "file_path": "/path/to/snapshot/file.json"
  }
}
```

#### `GET /sandbox_snapshot/{snapshot_id}`
Get a specific snapshot by ID.

**Parameters:**
- `snapshot_id` (path): The ID of the snapshot to retrieve

**Response:**
```json
{
  "success": true,
  "message": "Snapshot retrieved successfully",
  "data": {
    "timestamp": "2024-01-01T12:00:00Z",
    "snapshot_id": "snapshot_1704110400",
    "data": {...},
    "metadata": {...}
  }
}
```

#### `GET /sandbox_snapshots`
List all available snapshots.

**Response:**
```json
{
  "success": true,
  "message": "Found 10 snapshots",
  "data": {
    "snapshots": [
      {
        "snapshot_id": "snapshot_1704110400",
        "timestamp": "2024-01-01T12:00:00Z",
        "file_path": "/path/to/snapshot/file.json",
        "file_size": 2048000,
        "is_latest": true
      }
    ],
    "total_count": 10
  }
}
```

#### `POST /sandbox_snapshot/trigger`
Manually trigger a snapshot creation.

**Response:**
```json
{
  "success": true,
  "message": "Snapshot created successfully",
  "data": {
    "timestamp": "2024-01-01T12:00:00Z",
    "snapshot_id": "snapshot_1704110400",
    "data": {...},
    "metadata": {...}
  }
}
```

## Background Tasks

### Automatic Snapshot Creation
- Snapshots are automatically created every 1 minute
- Background task runs continuously while the server is active
- Snapshots are stored in the `snapshots/` directory
- Old snapshots are automatically cleaned up (configurable retention period)

### Transcript Generation
- Transcripts are generated when audio chunks are processed
- Files are saved in the `transcripts/{session_id}/` directory
- Filename format: `{session_id}_{chunk_id}.txt`

## File Structure

```
websocket/
├── transcripts/           # Transcript files
│   └── {session_id}/
│       └── {session_id}_{chunk_id}.txt
├── snapshots/            # Snapshot files
│   └── snapshot_YYYYMMDD_HHMMSS.json
├── transcript_service.py  # Transcript service module
├── snapshot_service.py    # Snapshot service module
└── main.py               # Updated FastAPI application
```

## Error Handling

All endpoints include comprehensive error handling:

- **400 Bad Request**: Invalid parameters
- **404 Not Found**: Resource not found (chunk, session, snapshot)
- **500 Internal Server Error**: Server-side errors

Error responses follow this format:
```json
{
  "success": false,
  "message": "Error description",
  "error": "Detailed error message"
}
```

## Logging

Comprehensive logging is implemented for debugging:

- All operations are logged with appropriate levels (INFO, WARNING, ERROR)
- Logs include timestamps, operation details, and error information
- Logs help track the flow of transcript generation and snapshot creation

## Testing

A test script is provided (`test_endpoints.py`) to verify all endpoints work correctly:

```bash
python test_endpoints.py
```

The test script will:
1. Check server health
2. Test all new endpoints
3. Verify existing endpoints still work
4. Provide a summary of test results

## Usage Examples

### Generate Transcript for Audio Chunk
```bash
curl -X POST "http://localhost:8000/process_chunk/123?session_id=session-456"
```

### Get Latest Sandbox Snapshot
```bash
curl "http://localhost:8000/sandbox_snapshot"
```

### Trigger Manual Snapshot
```bash
curl -X POST "http://localhost:8000/sandbox_snapshot/trigger"
```

### List All Snapshots
```bash
curl "http://localhost:8000/sandbox_snapshots"
```

## Configuration

### Snapshot Interval
The snapshot interval can be configured in `snapshot_service.py`:
```python
self.snapshot_interval = 60  # 1 minute in seconds
```

### Cleanup Settings
Old snapshots are cleaned up automatically. The retention period can be configured:
```python
await snapshot_service.cleanup_old_snapshots(keep_days=7)  # Keep 7 days
```

## Dependencies

The new functionality requires the existing dependencies plus:
- `pathlib` (built-in)
- `json` (built-in)
- `asyncio` (built-in)

No additional external dependencies are required.

## Security Considerations

- All file operations are performed in designated directories
- Input validation is performed on all parameters
- Error messages don't expose sensitive information
- File paths are sanitized to prevent directory traversal attacks

## Performance Considerations

- Snapshots are created asynchronously to avoid blocking the main event loop
- File I/O operations are performed efficiently
- Memory usage is optimized for large datasets
- Background tasks are designed to be lightweight and non-blocking
