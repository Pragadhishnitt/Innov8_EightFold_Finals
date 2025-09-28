#!/usr/bin/env python3
"""
Integration test script that starts the services and tests the new endpoints.
"""

import subprocess
import time
import requests
import json
import logging
import signal
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_endpoint(url, method="GET", data=None):
    """Test a single endpoint"""
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        logger.info(f"{method} {url} - Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Response: {json.dumps(result, indent=2)[:200]}...")
            return True
        else:
            logger.error(f"Error response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return False

def main():
    """Run integration test"""
    logger.info("🧪 Starting integration test...")
    
    # Start the main service
    logger.info("🚀 Starting main service...")
    main_process = subprocess.Popen([
        sys.executable, "main.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for service to start
    logger.info("⏳ Waiting for main service to start...")
    time.sleep(10)
    
    try:
        # Test the new endpoints
        logger.info("🔍 Testing new endpoints...")
        
        tests = [
            ("Health Check", "GET", "http://localhost:8000/health"),
            ("Sandbox Snapshot", "GET", "http://localhost:8000/sandbox_snapshot"),
            ("List Snapshots", "GET", "http://localhost:8000/sandbox_snapshots"),
            ("Trigger Snapshot", "POST", "http://localhost:8000/sandbox_snapshot/trigger"),
            ("Process Chunk (dummy)", "POST", "http://localhost:8000/process_chunk/test-123"),
            ("Sessions List", "GET", "http://localhost:8000/sessions"),
        ]
        
        results = {}
        for test_name, method, url in tests:
            logger.info(f"\n📋 Testing: {test_name}")
            success = test_endpoint(url, method)
            results[test_name] = "PASS" if success else "FAIL"
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("📊 Test Results:")
        for test_name, result in results.items():
            status_emoji = "✅" if result == "PASS" else "❌"
            logger.info(f"  {status_emoji} {test_name}: {result}")
        
        passed = sum(1 for r in results.values() if r == "PASS")
        total = len(results)
        logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! Integration successful!")
        else:
            logger.info("⚠️  Some tests failed. Check the logs above.")
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
    finally:
        # Cleanup
        logger.info("🛑 Stopping main service...")
        main_process.terminate()
        main_process.wait()
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    main()
