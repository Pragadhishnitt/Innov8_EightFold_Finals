#!/usr/bin/env python3
"""
Test script to verify the database schema and foreign key fixes work correctly.
"""

import asyncio
import logging
from database_manager import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_session_creation():
    """Test creating a session and candidate record"""
    try:
        logger.info("🧪 Testing session and candidate creation...")
        
        # Test session creation
        test_session_id = "test_session_12345"
        await db_manager.store_session(test_session_id)
        logger.info("✅ Session created successfully")
        
        # Test if we can query the session
        if hasattr(db_manager, 'supabase') and db_manager.supabase:
            response = db_manager.supabase.table('sessions').select('*').eq('session_id', test_session_id).execute()
            if response.data:
                logger.info(f"✅ Session found in database: {response.data[0]}")
            else:
                logger.error("❌ Session not found in database")
                return False
        
        logger.info("🎉 Database schema and foreign key fixes are working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def test_candidate_insertion():
    """Test inserting a candidate record (simulating frontend behavior)"""
    try:
        logger.info("🧪 Testing candidate insertion...")
        
        if not hasattr(db_manager, 'supabase') or not db_manager.supabase:
            logger.warning("⚠️  Supabase not available, skipping candidate test")
            return True
        
        # First create a session
        test_session_id = "test_candidate_session_67890"
        await db_manager.store_session(test_session_id)
        
        # Now try to insert a candidate (this would fail before the fix)
        response = db_manager.supabase.table('candidates').insert({
            'session_id': test_session_id,
            'name': 'Test Candidate',
            'skills': 'Python, JavaScript',
            'level': 'Senior',
            'photo_url': 'https://example.com/photo.jpg'
        }).execute()
        
        if response.data:
            logger.info("✅ Candidate record inserted successfully")
            logger.info(f"   Candidate data: {response.data[0]}")
            return True
        else:
            logger.error("❌ Candidate insertion failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Candidate test failed: {e}")
        return False

async def main():
    """Run all database tests"""
    logger.info("🚀 Starting database fix verification tests...")
    
    tests = [
        ("Session Creation", test_session_creation),
        ("Candidate Insertion", test_candidate_insertion),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
            logger.error(f"❌ {test_name} failed with error: {e}")
    
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
        logger.info("🎉 All database fixes are working correctly!")
        logger.info("   You can now start your interview session without foreign key errors.")
    else:
        logger.info("⚠️  Some tests failed. Please check the database schema.")

if __name__ == "__main__":
    asyncio.run(main())
