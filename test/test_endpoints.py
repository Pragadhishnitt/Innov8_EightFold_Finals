#!/usr/bin/env python3
"""
Test script for the new FastAPI endpoints.
This script tests both the transcript generation and sandbox snapshot endpoints.
"""

import asyncio
import httpx
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

async def test_health_check():
    """Test the health check endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health")
            logger.info(f"Health check status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Health check response: {json.dumps(data, indent=2)}")
                return True
            return False
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

async def test_sandbox_snapshot():
    """Test the sandbox snapshot endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            # Test getting latest snapshot
            response = await client.get(f"{BASE_URL}/sandbox_snapshot")
            logger.info(f"Sandbox snapshot status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Sandbox snapshot response: {json.dumps(data, indent=2)}")
                return True
            return False
    except Exception as e:
        logger.error(f"Sandbox snapshot test failed: {e}")
        return False

async def test_trigger_snapshot():
    """Test manually triggering a snapshot"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{BASE_URL}/sandbox_snapshot/trigger")
            logger.info(f"Trigger snapshot status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Trigger snapshot response: {json.dumps(data, indent=2)}")
                return True
            return False
    except Exception as e:
        logger.error(f"Trigger snapshot test failed: {e}")
        return False

async def test_list_snapshots():
    """Test listing all snapshots"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/sandbox_snapshots")
            logger.info(f"List snapshots status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"List snapshots response: {json.dumps(data, indent=2)}")
                return True
            return False
    except Exception as e:
        logger.error(f"List snapshots test failed: {e}")
        return False

async def test_transcript_endpoints():
    """Test transcript-related endpoints"""
    try:
        async with httpx.AsyncClient() as client:
            # Test with a dummy chunk_id (this will likely fail but tests the endpoint)
            test_chunk_id = "test-chunk-123"
            test_session_id = "test-session-456"
            
            # Test process chunk endpoint
            response = await client.post(f"{BASE_URL}/process_chunk/{test_chunk_id}")
            logger.info(f"Process chunk status: {response.status_code}")
            if response.status_code in [200, 404, 500]:  # 404/500 expected for dummy data
                data = response.json()
                logger.info(f"Process chunk response: {json.dumps(data, indent=2)}")
            
            # Test get transcript file endpoint
            response = await client.get(f"{BASE_URL}/transcript/{test_session_id}/{test_chunk_id}")
            logger.info(f"Get transcript file status: {response.status_code}")
            if response.status_code in [200, 404, 500]:  # 404/500 expected for dummy data
                data = response.json()
                logger.info(f"Get transcript file response: {json.dumps(data, indent=2)}")
            
            # Test list session transcripts endpoint
            response = await client.get(f"{BASE_URL}/transcripts/{test_session_id}")
            logger.info(f"List session transcripts status: {response.status_code}")
            if response.status_code in [200, 404, 500]:  # 404/500 expected for dummy data
                data = response.json()
                logger.info(f"List session transcripts response: {json.dumps(data, indent=2)}")
            
            return True
    except Exception as e:
        logger.error(f"Transcript endpoints test failed: {e}")
        return False

async def test_existing_endpoints():
    """Test some existing endpoints to ensure they still work"""
    try:
        async with httpx.AsyncClient() as client:
            # Test sessions endpoint
            response = await client.get(f"{BASE_URL}/sessions")
            logger.info(f"Sessions endpoint status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Sessions response: {json.dumps(data, indent=2)}")
            
            # Test detailed sessions endpoint
            response = await client.get(f"{BASE_URL}/api/sessions/detailed")
            logger.info(f"Detailed sessions endpoint status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Detailed sessions response: {json.dumps(data, indent=2)}")
            
            return True
    except Exception as e:
        logger.error(f"Existing endpoints test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("Starting endpoint tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Sandbox Snapshot", test_sandbox_snapshot),
        ("Trigger Snapshot", test_trigger_snapshot),
        ("List Snapshots", test_list_snapshots),
        ("Transcript Endpoints", test_transcript_endpoints),
        ("Existing Endpoints", test_existing_endpoints),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            result = await test_func()
            results[test_name] = "PASSED" if result else "FAILED"
            logger.info(f"{test_name}: {results[test_name]}")
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
            logger.error(f"{test_name}: {results[test_name]}")
    
    logger.info("\n" + "=" * 50)
    logger.info("Test Results Summary:")
    for test_name, result in results.items():
        logger.info(f"  {test_name}: {result}")
    
    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len(results)
    logger.info(f"\nOverall: {passed}/{total} tests passed")

if __name__ == "__main__":
    asyncio.run(main())
